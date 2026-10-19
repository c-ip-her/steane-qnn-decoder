# QNN and Error Correction: Neural-Network Decoding of the Seven-Qubit Steane Code

**Problem statement:** Train a quantum or classical network to decode Steane syndromes, and compare with the lookup table.  
**Execution date:** 19 October 2026 · **Evaluation split:** 1,500 saved test trajectories · **Runtime:** CPU

## 1. Problem

Quantum error-correcting codes infer likely physical errors from stabilizer measurement outcomes called syndromes. The seven-qubit Steane code is a CSS code that treats bit-flip and phase-flip components using binary parity checks. Syndrome measurements may themselves be noisy, so the most recent observed syndrome can differ from the syndrome associated with the accumulated data error. This project compares a fixed reference lookup rule with learned classical decoders. A Qiskit quantum neural network (QNN) was part of the intended comparison, but could not be executed in this runtime because Qiskit packages were absent and the package index could not be reached; no QNN result is reported.

## 2. Method

The saved dataset was reused without regeneration: 6,000 training, 1,200 validation, and 1,500 test trajectories. Each trajectory contains three rounds of six observed syndrome bits. Data-error probabilities were p = 0.02, 0.06, 0.12; syndrome measurement-bit flip probabilities were q = 0.00, 0.03, 0.08. The fixed data split seeds are 2026, 2027, and 2028. Per-round non-identity Pauli errors are sampled independently per qubit, with X/Y/Z chosen uniformly when an error occurs. Errors accumulate across rounds; there is no intermediate correction.

The six-bit convention is [H x (3 bits), H z (3 bits)], where H is the supplied 3×7 parity-check matrix and arithmetic is modulo 2. Labels are reference single-qubit labels: 0 for zero syndrome and 1–7 for the matching column of H. These labels do not uniquely identify arbitrary multi-qubit errors.

- **Lookup:** deterministic reference mapping applied to the latest observed six-bit syndrome.
- **MLP:** 6 → 32 → 16 logits with Tanh, interpreted as two 8-class heads; latest round only.
- **GRU:** GRU with input size 6 and hidden size 24, followed by 16 logits; all three rounds.
- **Training:** PyTorch, Adam, cross-entropy loss, validation-based best checkpoint selection; 15 epochs for each learned classical model in this run.
- **Evaluation:** exact match of both component labels, plus X and Z component accuracies. Inference latency is the mean of 100 single-sample forward passes after 10 warm-ups. The execution environment was CPU-only for this run.

## 3. Results

| Model | Input rounds | Exact label accuracy | X accuracy | Z accuracy | Test loss | Training time |
|---|---:|---:|---:|---:|---:|---:|
| Lookup | 1 | 80.80% | 89.40% | 89.87% | N/A | N/A |
| MLP | 1 | 80.80% | 89.40% | 89.87% | 0.511 | 1.62 s |
| GRU | 3 | 82.07% | 89.87% | 90.60% | 0.396 | 2.63 s |
| QNN | — | Not run | Not run | Not run | Not run | Not run |

The GRU achieved 82.07% exact label accuracy, compared with 80.80% for lookup and 80.80% for the MLP. The GRU's improvement over lookup was about 1.27 percentage points. It also receives three rounds, while lookup and MLP receive only one, so this result does not isolate architecture from information availability. The MLP matched the lookup exact accuracy on this test split. The GRU test loss was 0.396, and the MLP test loss was 0.511.

The measured single-sample inference latency was 0.019 ms for the MLP and 0.151 ms for the GRU in this CPU runtime. These micro-benchmarks depend on the host and software stack and should not be treated as universal performance numbers.

## 4. Honest Limits

1. The reference lookup table is not a general optimal decoder for multi-qubit errors. Distinct physical errors may share a syndrome.
2. The target is a pair of reference correction labels derived from the final true syndrome. Label accuracy is not the same as demonstrated physical recovery success or logical-error suppression.
3. The GRU uses all three observed rounds; lookup and MLP use only the latest round. Their scores are therefore not a strictly equal-input architecture comparison.
4. The QNN was **not executed**. The environment lacked Qiskit and Qiskit Machine Learning, and installation failed because the package index could not be reached. There are no QNN benchmark numbers and no quantum-hardware experiment.
5. A logical-error rate was not measured because this run did not perform a complete encoded-state recovery and logical-observable evaluation.
6. The dataset is synthetic, the classical models were trained for 15 epochs, and the results depend on the specified noise model and split.

**Conclusion.** On the executed test split, the three-round GRU performed best among the completed approaches, while the MLP matched the fixed lookup table in exact label accuracy. The requested QNN comparison remains incomplete and must be run in an environment with compatible Qiskit dependencies. These results do not establish quantum advantage.
