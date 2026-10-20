# Steane Code Error Decoder: Lookup, MLP, GRU, and Qiskit QNN

A small, reproducible learning project comparing a fixed syndrome lookup rule, a single-round MLP, a three-round GRU, and a trainable Qiskit quantum neural network (QNN) for a seven-qubit Steane-code-inspired syndrome-decoding task.

## Project objective

The project demonstrates how noisy syndrome measurements can be turned into correction labels, how classical neural networks learn from saved data, and how a trainable variational circuit can be integrated with PyTorch through Qiskit Machine Learning. It is a simulator experiment, not a claim of quantum advantage.

## Important interpretation

This is a **reference-label classification baseline**, not a full fault-tolerant decoder. The label for each three-bit syndrome is the matching single-qubit column of `H` (or label 0 for zero syndrome). Multi-qubit errors can share syndromes, so these labels do not uniquely identify the physical error. The logical-error rate is not reported because this repository does not implement a complete encoded-state recovery and logical-observable simulation.

The six-bit feature order is fixed everywhere as `[H @ x_error (3 bits), H @ z_error (3 bits)]`, with arithmetic modulo 2. This names the syndrome components by the error component being decoded; the physical CSS stabilizer checks anticommute with the opposite Pauli component. The same convention is used for data, labels, lookup, and learned models.

## Noise model and dataset

Each trajectory has three rounds. In each round, every data qubit independently receives a non-identity Pauli error with probability `p`; when an error occurs, X, Y, and Z are sampled uniformly. Errors accumulate by XOR in the binary X/Z representation. There is no intermediate correction. Each of the six syndrome measurement bits is independently flipped with probability `q`.

- Data-error probabilities: `0.02`, `0.06`, `0.12`
- Measurement-bit flip probabilities: `0.00`, `0.03`, `0.08`
- Train/validation/test sizes: 6000 / 1200 / 1500
- Fixed seeds: 2026 / 2027 / 2028

The generated files are compressed NumPy archives in `data/`. The generation function refuses to overwrite existing splits unless `overwrite=True` is explicitly passed.

## Models

- **Lookup:** deterministic single-qubit reference mapping applied to the latest observed syndrome. It receives no hidden true syndrome.
- **MLP:** latest observed six-bit syndrome → `Linear(6, 32)` → `Tanh` → `Linear(32, 16)`, reshaped into two 8-class predictions.
- **GRU:** all three observed six-bit syndromes → GRU with hidden size 24 → 16 logits, reshaped into two 8-class predictions.
- **QNN:** six-qubit parameterized Qiskit circuit with input rotations, trainable RY/RZ rotations, nearest-neighbour RZZ entanglers, and local Z plus neighbouring ZZ expectation values. A small classical output layer maps quantum features to the two correction labels. The notebook uses a local `StatevectorEstimator` for exact expectation values.

The MLP sees one round while the GRU sees three; this information difference must be considered when interpreting their scores.

## Installation

Python 3.11 is recommended. Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Notebook execution order

Run from the repository root:

1. `notebooks/01_generate_data.ipynb` — test the syndrome functions, create splits if missing, and inspect examples.
2. `notebooks/02_lookup_mlp_gru.ipynb` — evaluate lookup, train MLP and GRU, and save classical checkpoints.
3. `notebooks/03_qiskit_qnn.ipynb` — construct and validate the Qiskit circuit, train the QNN, and save its checkpoint.
4. `notebooks/04_evaluation.ipynb` — compare all models and save metrics/figures.

`main.py` contains the shared syndrome, data, MLP, GRU, training, and metric functions. Run its basic checks with:

```bash
python -c "from main import run_self_tests; print(run_self_tests())"
```

If you intentionally want to recreate data, first back up or remove the existing `.npz` files, or call `save_splits(overwrite=True)` from a notebook cell. Training notebooks load the saved files and do not regenerate them.

## Results and limitations

The notebooks write observed results to `results/metrics.json` and plots to `results/figures/`. No benchmark numbers are pre-filled in this README. Results depend on the actual run and should be discussed with the generated metrics.

The lookup and neural models predict reference labels, not a full physical recovery operation. The QNN uses an exact local simulator, not quantum hardware. Training a QNN is not evidence of quantum advantage; its accuracy and runtime should be compared empirically with the classical baselines.
