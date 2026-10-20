"""
run_pipeline.py
───────────────
End-to-end runner for the Steane-code syndrome-decoding project.
Reproduces all four notebook steps in order:
  1. Generate data (or reuse existing)
  2. Lookup baseline, train MLP and GRU, save classical_metrics.json
  3. Build and train hybrid Qiskit QNN, save qnn.pt
  4. Evaluate all models, save metrics.json, test_comparison.csv, noise_breakdown.csv, figures

Run from the steane-qnn/ directory:
    python run_pipeline.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from main import (
    run_self_tests,
    save_splits,
    load_split,
    MLPDecoder,
    GRUDecoder,
    train_model,
    predict_labels,
    evaluate_predictions,
    lookup_predict,
    batch_loss,
    save_json,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGS_DIR    = RESULTS_DIR / "figures"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Syndrome self-tests and data generation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1 – Self-tests and data generation")
print("=" * 70)

print(run_self_tests())

expected = [DATA_DIR / "train.npz", DATA_DIR / "validation.npz", DATA_DIR / "test.npz"]
if all(p.exists() for p in expected):
    print("Dataset files already exist – reusing without regeneration.")
else:
    save_splits(DATA_DIR, overwrite=False)

train      = load_split(DATA_DIR / "train.npz")
validation = load_split(DATA_DIR / "validation.npz")
test       = load_split(DATA_DIR / "test.npz")

print(f"Train: {len(train['observed'])} | Validation: {len(validation['observed'])} | Test: {len(test['observed'])}")

y_test = np.stack([test["labels_x"], test["labels_z"]], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Lookup baseline, MLP, GRU
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2 – Lookup baseline, MLP, GRU")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Lookup
lookup_scores = evaluate_predictions(lookup_predict(test["observed"]), y_test)
print("Lookup baseline:", lookup_scores)

# MLP
torch.manual_seed(2026)
mlp, mlp_history = train_model(
    MLPDecoder(), "mlp", train, validation, MODELS_DIR / "mlp.pt",
    epochs=25, batch_size=128, learning_rate=1e-3, device=device,
)

plt.figure(figsize=(7, 4))
plt.plot(mlp_history["train_loss"], label="train")
plt.plot(mlp_history["validation_loss"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("MLP training history")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / "mlp_loss.png", dpi=150)
plt.close()
print("Saved results/figures/mlp_loss.png")

# GRU
torch.manual_seed(2027)
gru, gru_history = train_model(
    GRUDecoder(hidden_size=24), "gru", train, validation, MODELS_DIR / "gru.pt",
    epochs=25, batch_size=128, learning_rate=1e-3, device=device,
)

plt.figure(figsize=(7, 4))
plt.plot(gru_history["train_loss"], label="train")
plt.plot(gru_history["validation_loss"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("GRU training history")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / "gru_loss.png", dpi=150)
plt.close()
print("Saved results/figures/gru_loss.png")

# Load best checkpoints and evaluate on test
mlp_checkpoint = torch.load(MODELS_DIR / "mlp.pt", map_location=device, weights_only=False)
gru_checkpoint = torch.load(MODELS_DIR / "gru.pt", map_location=device, weights_only=False)
mlp.load_state_dict(mlp_checkpoint["model_state_dict"])
gru.load_state_dict(gru_checkpoint["model_state_dict"])

mlp_features = test["observed"][:, -1, :].astype(np.float32)
gru_features = test["observed"].astype(np.float32)
mlp_scores = evaluate_predictions(predict_labels(mlp, mlp_features, device=device), y_test)
gru_scores = evaluate_predictions(predict_labels(gru, gru_features, device=device), y_test)
print("MLP test metrics:", mlp_scores)
print("GRU test metrics:", gru_scores)

save_json(
    {
        "lookup": lookup_scores,
        "mlp": mlp_scores,
        "gru": gru_scores,
        "mlp_training_seconds": mlp_history["training_seconds"],
        "gru_training_seconds": gru_history["training_seconds"],
        "mlp_history": mlp_history,
        "gru_history": gru_history,
    },
    RESULTS_DIR / "classical_metrics.json",
)
print("Saved results/classical_metrics.json")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Qiskit hybrid QNN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 – Qiskit hybrid QNN")
print("=" * 70)

QNN_AVAILABLE = False
qnn_logits    = None
qnn_history   = {}

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector

    torch.manual_seed(2026)
    np.random.seed(2026)

    N_QUBITS = 6
    N_LAYERS = 2
    input_params  = ParameterVector("x",     N_QUBITS)
    weight_params = ParameterVector("theta",  N_LAYERS * N_QUBITS * 2)

    qc = QuantumCircuit(N_QUBITS)
    for qubit in range(N_QUBITS):
        qc.ry(np.pi * input_params[qubit], qubit)

    cursor = 0
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qc.ry(weight_params[cursor], qubit); cursor += 1
            qc.rz(weight_params[cursor], qubit); cursor += 1
        for qubit in range(N_QUBITS - 1):
            qc.rzz(0.5, qubit, qubit + 1)

    def pauli_string(occupied_qubits, n_qubits=6):
        chars = ["I"] * n_qubits
        for qubit, pauli in occupied_qubits:
            chars[n_qubits - 1 - qubit] = pauli
        return "".join(chars)

    observables  = [SparsePauliOp(pauli_string([(q, "Z")])) for q in range(N_QUBITS)]
    observables += [SparsePauliOp(pauli_string([(q, "Z"), (q + 1, "Z")])) for q in range(N_QUBITS - 1)]

    print("Circuit:")
    try:
        print(qc.draw("text"))
    except Exception:
        print(repr(qc))
    print(f"Number of expectation-value features: {len(observables)}")

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=StatevectorEstimator(),
        input_params=list(input_params),
        weight_params=list(weight_params),
        observables=observables,
        input_gradients=True,
    )
    initial_weights = np.random.uniform(-0.1, 0.1, len(weight_params))
    quantum_layer   = TorchConnector(qnn, initial_weights=initial_weights)

    class HybridQNN(nn.Module):
        def __init__(self, quantum_layer):
            super().__init__()
            self.quantum_layer = quantum_layer
            self.head = nn.Linear(len(observables), 16)

        def forward(self, x):
            features = self.quantum_layer(x)
            return self.head(features).view(-1, 2, 8)

    qnn_model = HybridQNN(quantum_layer).to(device)

    # Sanity-check forward pass + gradients
    probe_x = torch.tensor(train["observed"][:4, -1, :], dtype=torch.float32, device=device)
    probe_y = torch.tensor(
        np.stack([train["labels_x"][:4], train["labels_z"][:4]], axis=1), dtype=torch.long, device=device
    )
    probe_out  = qnn_model(probe_x)
    probe_loss = batch_loss(probe_out, probe_y)
    probe_loss.backward()
    assert probe_out.shape == (4, 2, 8)
    assert torch.isfinite(probe_loss)
    assert qnn_model.quantum_layer.weight.grad is not None
    print(f"Sanity check passed – output shape {tuple(probe_out.shape)}, finite loss {float(probe_loss):.4f}")

    # Training
    x_train_all = train["observed"][:, -1, :].astype(np.float32)
    y_train_all = np.stack([train["labels_x"], train["labels_z"]], axis=1).astype(np.int64)
    x_val       = validation["observed"][:, -1, :].astype(np.float32)
    y_val       = np.stack([validation["labels_x"], validation["labels_z"]], axis=1).astype(np.int64)

    subset_size       = min(512, len(x_train_all))
    x_train_small     = x_train_all[:subset_size]
    y_train_small     = y_train_all[:subset_size]
    qnn_train_loader  = DataLoader(
        TensorDataset(torch.from_numpy(x_train_small), torch.from_numpy(y_train_small)),
        batch_size=16, shuffle=True,
    )

    optimizer     = torch.optim.Adam(qnn_model.parameters(), lr=0.01)
    qnn_history   = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
    best_val_acc  = -1.0
    qnn_started   = time.perf_counter()

    QNN_EPOCHS = 8
    for epoch in range(QNN_EPOCHS):
        qnn_model.train()
        batch_losses = []
        for xb, yb in qnn_train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = qnn_model(xb)
            loss   = batch_loss(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        qnn_model.eval()
        with torch.no_grad():
            val_logits_list = []
            for start in range(0, len(x_val), 32):
                xb = torch.tensor(x_val[start:start + 32], dtype=torch.float32, device=device)
                val_logits_list.append(qnn_model(xb).cpu().numpy())
            val_logits_arr = np.concatenate(val_logits_list)
            val_loss       = float(
                batch_loss(torch.tensor(val_logits_arr), torch.tensor(y_val)).item()
            )

        val_metrics = evaluate_predictions(val_logits_arr, y_val)
        epoch_train_loss = float(np.mean(batch_losses))
        qnn_history["train_loss"].append(epoch_train_loss)
        qnn_history["validation_loss"].append(val_loss)
        qnn_history["validation_accuracy"].append(val_metrics["correction_label_accuracy"])
        print(
            f"QNN epoch {epoch + 1}/{QNN_EPOCHS}: "
            f"train_loss={epoch_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_exact={val_metrics['correction_label_accuracy']:.3f}"
        )

        if val_metrics["correction_label_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["correction_label_accuracy"]
            torch.save(
                {
                    "model_state_dict": qnn_model.state_dict(),
                    "n_layers": N_LAYERS,
                    "n_qubits": N_QUBITS,
                    "history": qnn_history,
                    "best_validation_accuracy": best_val_acc,
                    "input_params": [str(p) for p in input_params],
                    "weight_params": [str(p) for p in weight_params],
                },
                MODELS_DIR / "qnn.pt",
            )

    qnn_history["training_seconds"] = time.perf_counter() - qnn_started
    best_ckpt = torch.load(MODELS_DIR / "qnn.pt", map_location="cpu", weights_only=False)
    best_ckpt["history"] = qnn_history
    torch.save(best_ckpt, MODELS_DIR / "qnn.pt")

    plt.figure(figsize=(7, 4))
    plt.plot(qnn_history["train_loss"], label="train")
    plt.plot(qnn_history["validation_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("QNN training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "qnn_loss.png", dpi=150)
    plt.close()
    print("Saved results/figures/qnn_loss.png")
    print(f"QNN training done in {qnn_history['training_seconds']:.1f}s – best val accuracy: {best_val_acc:.3f}")
    QNN_AVAILABLE = True

except ImportError as exc:
    print(f"[WARNING] Qiskit packages not available ({exc}). Skipping QNN step.")
    QNN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Evaluate all models and write results
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 – Evaluation and results")
print("=" * 70)

results = {}
results["lookup"] = evaluate_predictions(lookup_predict(test["observed"]), y_test)

mlp_ckpt = torch.load(MODELS_DIR / "mlp.pt", map_location=device, weights_only=False)
gru_ckpt = torch.load(MODELS_DIR / "gru.pt", map_location=device, weights_only=False)
mlp2 = MLPDecoder().to(device)
gru2 = GRUDecoder(hidden_size=24).to(device)
mlp2.load_state_dict(mlp_ckpt["model_state_dict"])
gru2.load_state_dict(gru_ckpt["model_state_dict"])
results["mlp"] = evaluate_predictions(predict_labels(mlp2, mlp_features, device=device), y_test)
results["gru"] = evaluate_predictions(predict_labels(gru2, gru_features, device=device), y_test)

def logits_for(model, features, batch_size=64):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(xb).cpu())
    return torch.cat(chunks, dim=0)

mlp_logits_t = logits_for(mlp2, mlp_features)
gru_logits_t = logits_for(gru2, gru_features)
labels_t     = torch.tensor(y_test, dtype=torch.long)

learned_losses = {
    "mlp": float(batch_loss(mlp_logits_t, labels_t).item()),
    "gru": float(batch_loss(gru_logits_t, labels_t).item()),
}
training_times = {
    "mlp": float(mlp_ckpt.get("history", {}).get("training_seconds", float("nan"))),
    "gru": float(gru_ckpt.get("history", {}).get("training_seconds", float("nan"))),
}

qnn_inference_seconds = float("nan")
qnn_logits            = None

if QNN_AVAILABLE:
    qnn2_input_params  = ParameterVector("x",     N_QUBITS)
    qnn2_weight_params = ParameterVector("theta",  N_LAYERS * N_QUBITS * 2)
    qc2 = QuantumCircuit(N_QUBITS)
    for qubit in range(N_QUBITS):
        qc2.ry(np.pi * qnn2_input_params[qubit], qubit)
    cursor = 0
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qc2.ry(qnn2_weight_params[cursor], qubit); cursor += 1
            qc2.rz(qnn2_weight_params[cursor], qubit); cursor += 1
        for qubit in range(N_QUBITS - 1):
            qc2.rzz(0.5, qubit, qubit + 1)
    obs2  = [SparsePauliOp(pauli_string([(q, "Z")])) for q in range(N_QUBITS)]
    obs2 += [SparsePauliOp(pauli_string([(q, "Z"), (q + 1, "Z")])) for q in range(N_QUBITS - 1)]
    qnn2  = EstimatorQNN(
        circuit=qc2, estimator=StatevectorEstimator(),
        input_params=list(qnn2_input_params), weight_params=list(qnn2_weight_params),
        observables=obs2, input_gradients=True,
    )
    ql2       = TorchConnector(qnn2, initial_weights=np.zeros(len(qnn2_weight_params)))

    class HybridQNN2(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.quantum_layer = layer
            self.head = nn.Linear(len(obs2), 16)
        def forward(self, x):
            return self.head(self.quantum_layer(x)).view(-1, 2, 8)

    qnn_model2 = HybridQNN2(ql2).to(device)
    qnn_ckpt   = torch.load(MODELS_DIR / "qnn.pt", map_location=device, weights_only=False)
    qnn_model2.load_state_dict(qnn_ckpt["model_state_dict"])
    qnn_model2.eval()

    qnn_logits_list = []
    inf_started     = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(mlp_features), 16):
            xb = torch.tensor(mlp_features[start:start + 16], dtype=torch.float32, device=device)
            qnn_logits_list.append(qnn_model2(xb).cpu().numpy())
    qnn_inference_seconds = time.perf_counter() - inf_started
    qnn_logits = np.concatenate(qnn_logits_list)
    results["qnn"] = evaluate_predictions(qnn_logits, y_test)
    print("QNN test metrics:", results["qnn"])

    learned_losses["qnn"] = float(batch_loss(torch.tensor(qnn_logits), labels_t).item())
    training_times["qnn"]  = float(qnn_ckpt.get("history", {}).get("training_seconds", float("nan")))

# ── Comparison table ────────────────────────────────────────────────────────────
rows       = [{"model": name, **metrics} for name, metrics in results.items()]
comparison = pd.DataFrame(rows)
print("\nTest-set comparison:")
print(comparison.to_string(index=False))
comparison.to_csv(RESULTS_DIR / "test_comparison.csv", index=False)
print("Saved results/test_comparison.csv")

# ── Accuracy bar chart ──────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.bar(comparison["model"], comparison["correction_label_accuracy"])
plt.ylabel("Exact two-component label accuracy")
plt.title("Test accuracy by decoder")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGS_DIR / "test_accuracy.png", dpi=150)
plt.close()
print("Saved results/figures/test_accuracy.png")

# ── Noise breakdown ─────────────────────────────────────────────────────────────
group_rows = []
eval_pairs = [("lookup", lookup_predict(test["observed"]))]
if qnn_logits is not None:
    eval_pairs.append(("qnn", qnn_logits))

for model_name, pred in eval_pairs:
    predicted = pred.argmax(axis=-1)
    for p in sorted(np.unique(test["p"])):
        mask = test["p"] == p
        group_rows.append({
            "model": model_name, "group": "p", "value": float(p),
            "accuracy": float(np.mean(np.all(predicted[mask] == y_test[mask], axis=1))),
        })
    for q in sorted(np.unique(test["q"])):
        mask = test["q"] == q
        group_rows.append({
            "model": model_name, "group": "q", "value": float(q),
            "accuracy": float(np.mean(np.all(predicted[mask] == y_test[mask], axis=1))),
        })

group_df = pd.DataFrame(group_rows)
group_df.to_csv(RESULTS_DIR / "noise_breakdown.csv", index=False)
print("Saved results/noise_breakdown.csv")

# ── metrics.json (canonical results file) ──────────────────────────────────────
metrics_payload = {
    "test_metrics": results,
    "cross_entropy_loss": learned_losses,
    "training_seconds": training_times,
    "qnn_inference_seconds_for_test_set": qnn_inference_seconds,
    "notes": [
        "Lookup and MLP use the latest observed syndrome; GRU uses all three rounds.",
        "Labels are reference single-qubit labels derived from the final true syndrome.",
        "QNN uses a local exact StatevectorEstimator; no quantum hardware was used." if QNN_AVAILABLE
            else "QNN was skipped because qiskit or qiskit-machine-learning is not installed.",
        "No logical-error rate is reported because full recovery/logical-observable simulation is not implemented.",
    ],
}
save_json(metrics_payload, RESULTS_DIR / "metrics.json")
print("Saved results/metrics.json")

# ── Summary ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIPELINE COMPLETE – Summary")
print("=" * 70)
for name, m in results.items():
    print(
        f"  {name:8s}  exact={m['correction_label_accuracy']:.3f}  "
        f"X={m['x_component_accuracy']:.3f}  Z={m['z_component_accuracy']:.3f}"
    )
print("=" * 70)
print(f"\nAll results written to: {RESULTS_DIR}")
