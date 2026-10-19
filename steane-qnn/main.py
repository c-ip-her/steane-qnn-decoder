"""
Reusable functions for the Steane-code syndrome-decoding project.

Syndrome convention:
  syndrome_x = H @ x_error (mod 2)
  syndrome_z = H @ z_error (mod 2)
  observed sequence columns are [syndrome_x (3 bits), syndrome_z (3 bits)].
For a CSS code, the physical stabilizer checks couple to the opposite Pauli
component; here the component names indicate which error component is being
decoded. The convention is used consistently for labels and all models.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

H = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
], dtype=np.int64)

DATA_ERROR_PROBS = (0.02, 0.06, 0.12)
MEASUREMENT_ERROR_PROBS = (0.00, 0.03, 0.08)
N_ROUNDS = 3


def syndrome_component(error_bits):
    """Return the three-bit syndrome for one 7-bit component."""
    return (H @ np.asarray(error_bits, dtype=np.int64)) % 2


def calculate_syndrome(x_error, z_error):
    """Return six bits ordered as [H@x_error, H@z_error], each mod 2."""
    sx = syndrome_component(x_error)
    sz = syndrome_component(z_error)
    return np.concatenate([sx, sz]).astype(np.int64)


def syndrome_to_label(syndrome_bits):
    """
    Map a three-bit syndrome to a reference single-qubit correction label.
    Label 0 means no correction; labels 1..7 refer to qubits 0..6.
    This is a reference lookup, not a general optimal decoder for multi-qubit
    errors: different physical errors can share a syndrome.
    """
    syndrome_bits = np.asarray(syndrome_bits, dtype=np.int64).reshape(3)
    if not syndrome_bits.any():
        return 0
    for qubit in range(7):
        if np.array_equal(H[:, qubit], syndrome_bits):
            return qubit + 1
    raise ValueError(f"No single-qubit reference label for syndrome {syndrome_bits}")


def correction_labels(syndrome_bits):
    """Return (X-component label, Z-component label) from a six-bit syndrome."""
    syndrome_bits = np.asarray(syndrome_bits, dtype=np.int64).reshape(6)
    return (
        syndrome_to_label(syndrome_bits[:3]),
        syndrome_to_label(syndrome_bits[3:]),
    )


def sample_pauli_error(rng, p):
    """Sample independent per-qubit Pauli errors with total non-identity rate p."""
    draw = rng.random(7)
    pauli = rng.integers(1, 4, size=7)  # 1=X, 2=Y, 3=Z
    active = draw < p
    x_error = np.zeros(7, dtype=np.int64)
    z_error = np.zeros(7, dtype=np.int64)
    x_error[active] = np.isin(pauli[active], [1, 2]).astype(np.int64)
    z_error[active] = np.isin(pauli[active], [2, 3]).astype(np.int64)
    return x_error, z_error


def generate_split(n_samples, seed):
    """Generate trajectories with independently sampled p and q for each sample."""
    rng = np.random.default_rng(seed)
    observed = np.zeros((n_samples, N_ROUNDS, 6), dtype=np.int64)
    true_syndromes = np.zeros_like(observed)
    final_x = np.zeros((n_samples, 7), dtype=np.int64)
    final_z = np.zeros((n_samples, 7), dtype=np.int64)
    labels_x = np.zeros(n_samples, dtype=np.int64)
    labels_z = np.zeros(n_samples, dtype=np.int64)
    p_values = rng.choice(DATA_ERROR_PROBS, size=n_samples)
    q_values = rng.choice(MEASUREMENT_ERROR_PROBS, size=n_samples)

    for i in range(n_samples):
        x_acc = np.zeros(7, dtype=np.int64)
        z_acc = np.zeros(7, dtype=np.int64)
        for round_index in range(N_ROUNDS):
            dx, dz = sample_pauli_error(rng, float(p_values[i]))
            x_acc ^= dx
            z_acc ^= dz
            true = calculate_syndrome(x_acc, z_acc)
            noisy = true.copy()
            flips = rng.random(6) < q_values[i]
            noisy[flips] ^= 1
            true_syndromes[i, round_index] = true
            observed[i, round_index] = noisy

        final_x[i] = x_acc
        final_z[i] = z_acc
        labels_x[i], labels_z[i] = correction_labels(true_syndromes[i, -1])

    return {
        "observed": observed,
        "true_syndromes": true_syndromes,
        "final_x": final_x,
        "final_z": final_z,
        "labels_x": labels_x,
        "labels_z": labels_z,
        "p": p_values.astype(np.float32),
        "q": q_values.astype(np.float32),
    }


def save_splits(data_dir="data", sizes=(6000, 1200, 1500), seeds=(2026, 2027, 2028),
                overwrite=False):
    """Create compressed train/validation/test files, refusing accidental overwrite."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    names = ("train", "validation", "test")
    paths = [data_dir / f"{name}.npz" for name in names]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Dataset file(s) already exist. Reuse them, or call save_splits(..., "
            "overwrite=True) only when you intentionally want new data: "
            + ", ".join(str(p) for p in existing)
        )
    for name, size, seed, path in zip(names, sizes, seeds, paths):
        split = generate_split(size, seed)
        np.savez_compressed(path, **split)
        print(f"Saved {path} ({size} trajectories)")
    return paths


def load_split(path):
    """Load one saved NPZ split as a dictionary of NumPy arrays."""
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def run_self_tests():
    """Check all seven locations and X/Y/Z Pauli types."""
    for qubit in range(7):
        x = np.zeros(7, dtype=np.int64)
        z = np.zeros(7, dtype=np.int64)
        expected = H[:, qubit]
        # X error
        x[qubit] = 1
        assert np.array_equal(calculate_syndrome(x, z)[:3], expected)
        assert correction_labels(calculate_syndrome(x, z))[0] == qubit + 1
        x[qubit] = 0
        # Z error
        z[qubit] = 1
        assert np.array_equal(calculate_syndrome(x, z)[3:], expected)
        assert correction_labels(calculate_syndrome(x, z))[1] == qubit + 1
        # Y error
        x[qubit] = 1
        syn = calculate_syndrome(x, z)
        assert np.array_equal(syn[:3], expected)
        assert np.array_equal(syn[3:], expected)
    assert correction_labels(np.zeros(6, dtype=int)) == (0, 0)
    return "All syndrome and single-qubit reference-label tests passed."


class MLPDecoder(nn.Module):
    """Single-round decoder: 6 -> 32 -> 16 logits (8 X labels + 8 Z labels)."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
        )

    def forward(self, x):
        return self.network(x).view(-1, 2, 8)


class GRUDecoder(nn.Module):
    """Three-round decoder that summarizes a sequence with a GRU hidden state."""
    def __init__(self, hidden_size=24):
        super().__init__()
        self.gru = nn.GRU(input_size=6, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 16)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.output(hidden[-1]).view(-1, 2, 8)


def arrays_to_dataset(data, model_kind):
    """Prepare features and two correction labels for PyTorch."""
    if model_kind == "mlp":
        features = data["observed"][:, -1, :].astype(np.float32)
    elif model_kind == "gru":
        features = data["observed"].astype(np.float32)
    else:
        raise ValueError("model_kind must be 'mlp' or 'gru'")
    labels = np.stack([data["labels_x"], data["labels_z"]], axis=1).astype(np.int64)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


def batch_loss(logits, labels):
    return (nn.functional.cross_entropy(logits[:, 0, :], labels[:, 0]) +
            nn.functional.cross_entropy(logits[:, 1, :], labels[:, 1])) / 2.0


def predict_labels(model, features, batch_size=256, device="cpu"):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size],
                                    dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_predictions(predictions, labels):
    """Metrics use exact two-component match plus separate component accuracies."""
    predicted = predictions.argmax(axis=-1)
    labels = np.asarray(labels)
    return {
        "correction_label_accuracy": float(np.mean(np.all(predicted == labels, axis=1))),
        "x_component_accuracy": float(np.mean(predicted[:, 0] == labels[:, 0])),
        "z_component_accuracy": float(np.mean(predicted[:, 1] == labels[:, 1])),
    }


def train_model(model, model_kind, train_data, validation_data, model_path,
                epochs=25, batch_size=128, learning_rate=1e-3, device=None):
    """Train with Adam and keep the checkpoint with the best validation exact-match accuracy."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_ds = arrays_to_dataset(train_data, model_kind)
    val_ds = arrays_to_dataset(validation_data, model_kind)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
    best_accuracy = -1.0
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = batch_loss(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(features)
            seen += len(features)

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = batch_loss(logits, labels)
                val_loss_sum += float(loss.item()) * len(features)
                val_seen += len(features)
                all_predictions.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_metrics = evaluate_predictions(
            np.concatenate(all_predictions), np.concatenate(all_labels)
        )
        train_loss = running_loss / max(seen, 1)
        val_loss = val_loss_sum / max(val_seen, 1)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_metrics["correction_label_accuracy"])
        print(f"{model_kind.upper()} epoch {epoch + 1:02d}/{epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_exact={val_metrics['correction_label_accuracy']:.3f}")

        if val_metrics["correction_label_accuracy"] > best_accuracy:
            best_accuracy = val_metrics["correction_label_accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_kind": model_kind,
                "best_validation_accuracy": best_accuracy,
                "history": history,
            }, model_path)

    history["training_seconds"] = time.perf_counter() - started
    # Update the best checkpoint with the complete history and elapsed training time.
    best_checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    best_checkpoint["history"] = history
    torch.save(best_checkpoint, model_path)
    return model, history


def lookup_predict(observed_sequence):
    """
    Apply the fixed reference table to the latest observed six-bit syndrome.
    This deliberately uses only the same latest round as the MLP, not hidden truth.
    """
    observed_sequence = np.asarray(observed_sequence, dtype=np.int64)
    latest = observed_sequence[:, -1, :]
    predictions = np.zeros((len(latest), 2, 8), dtype=np.float32)
    for i, syndrome in enumerate(latest):
        lx, lz = correction_labels(syndrome)
        predictions[i, 0, lx] = 1.0
        predictions[i, 1, lz] = 1.0
    return predictions


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
