# Steane syndrome decoding with classical and quantum networks

Compare a minimum-weight lookup table, an MLP, a temporal GRU, and a six-qubit
quantum neural network. Measure gradient variance for 2–10 qubits and compare
random initialization, small-angle initialization, and local costs.

## Files

```text
main.py           All simulation, training, evaluation, plotting and verification code
requirements.txt  Pinned Python dependencies
README.md         Setup, commands and experiment details
results/          Recorded measurements, checkpoints, datasets and figures
WRITEUP.pdf       Final four-page report of the included reference run
```

The PDF is a standalone document. There is no report-generation code. Running
experiments does not change it. If you change the settings or results, revise
the report separately. The results summary is also a static reference document.

## Install

Reference environment: **Python 3.12.14**, Windows, CPU. Use Python 3.12.
Run these commands from the project folder. No cloud account or GPU is required.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is disabled, replace `python` in subsequent commands with
`.\.venv\Scripts\python.exe`.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify the included results

```bash
python main.py verify
```

Expected: 27 checkpoints, 13,824 gradient samples and 108 variance groups verified.
This reloads each model, recomputes its saved validation loss and accuracy, and
recomputes gradient variances from the raw samples. It does not retrain or check
every statement in the PDF.

## Run all experiments

```bash
python main.py all
```

This trains the classical and quantum models, evaluates noisy error trajectories,
trains on the complete syndrome table, measures gradient statistics, regenerates
figures, and verifies the outputs. Allow several minutes or longer depending on
your CPU. It overwrites the included measurements and figures: keep a reference
copy if you want to compare runs. Seeds and budgets are fixed; numerical results
should match closely, but cross-platform bitwise equality is not guaranteed.
Training times and inference latency depend on the computer.

## Train separately

Classical MLP and temporal GRU:

```bash
python main.py train --models mlp gru --out results/classical
```

Quantum models, with matched random and small-angle initialization and a frozen
quantum-feature control:

```bash
python main.py train --models qnn_random qnn_small qnn_small_frozen --out results/quantum
```

For the combined reference results used by the plots:

```bash
python main.py train --out results/main
python main.py train --task table --models mlp qnn_random qnn_small qnn_small_frozen --out results/table
python main.py plateau --qubits 2 3 4 5 6 7 8 9 10 --depths 1 3 6 --samples 128
python main.py plot
```

Plots are saved under `results/figures/`. Plotting and verification expect the
combined `results/main`, `results/table`, and `results/plateau` outputs.
Use `python main.py train --help` or `python main.py plateau --help` for options.

## Experiment

The [[7,1,3]] Steane code uses six stabilizer syndrome bits. The simulator
constructs logical states, applies Pauli errors, measures stabilizers and checks
recovery. Training datasets use the equivalent binary Pauli representation.
Three rounds accumulate independent depolarizing errors and syndrome-bit flips,
without intermediate recovery. Labels are minimum-weight corrections for the
true final syndrome.

The MLP sees the latest syndrome; the GRU sees three rounds. The hybrid QNN
encodes six syndrome bits, uses three layers of RY/RZ rotations and trainable RZZ
entanglers, measures six Z and five neighboring ZZ expectations, and feeds a
classical linear readout. Small-angle initialization uses Normal(0, 0.05);
the random baseline uses Uniform(-pi, pi). A frozen-feature model isolates the
effect of training only the classical readout.

Training uses Adam, learning rate 0.03, 120 updates and seeds 11, 22, 33.
The trajectory task uses 4,096 training and 1,024 validation trajectories.
Each noise-test setting uses 4,096 independent trajectories. Files retain
loss histories, timing, model weights, validation data, noise settings and
logical-error metrics, including raw recovery failure and ideal-final-cleanup
logical error. The complete-table task learns all 64 syndromes.

Gradient measurements use one fixed first-layer RY component, automatic
differentiation, 128 initializations per condition and bootstrap intervals.
Compare width 2–10, depth 1/3/6, local/global costs and both initializations.
Local cost is studied in this diagnostic, not as a replacement training loss
for the decoder. The mechanism/noise plot contains explicitly labeled analytical
examples; it is not a hardware experiment.

## Reference results and limits

Both trained QNNs achieve 100% accuracy on the complete table. Small-angle
initialization reaches cross-entropy <= 0.8 in 13.33 versus 19.33 mean updates
(31% fewer), but does not improve final loss (0.01365 versus 0.01336).
The MLP is faster. Single-round learned decoders match lookup logical-error rates
on the recorded test grid; the GRU has access to extra temporal information.

At depth six, descriptive log10-gradient-variance slopes are -0.524 for
random/global, -0.127 for small/global and -0.185 for random/local.
These finite-range results do not establish an asymptotic scaling law.

The table experiment is memorization, not held-out generalization. Logical error
after ideal final cleanup assumes an additional perfect recovery step. Only
three training seeds are used. There is no real quantum hardware execution,
faulty ancilla-extraction circuit or demonstrated quantum advantage. See
`results/RESULTS.md` and `WRITEUP.pdf` for the recorded comparisons.
