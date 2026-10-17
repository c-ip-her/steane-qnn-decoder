# Measured results

On the complete table, mean final CE is 0.01336 for random initialization and 0.01365 for small-angle initialization. Small-angle initialization does not have lower final loss. It reaches CE <= 0.8 in 13.33 versus 19.33 mean updates (31.0% fewer). Mean table accuracies are 100.0% (small) and 100.0% (random).

## Noisy trajectory comparison

| Model | Validation CE (mean +/- SD) | Training s (mean) | CPU latency ms (mean of medians) | Parameters |
|---|---:|---:|---:|---:|
| mlp | 0.39136 +/- 0.00214 | 0.59 | 0.042 | 752 |
| gru | 0.38626 +/- 0.00181 | 3.96 | 0.140 | 2704 |
| qnn_random | 0.39405 +/- 0.00268 | 4.11 | 3.837 | 243 |
| qnn_small | 0.39599 +/- 0.00131 | 4.09 | 3.896 | 243 |
| qnn_small_frozen | 0.39846 +/- 0.00127 | 2.74 | 4.157 | 192 |

## Complete table task

| Model | Final CE | Joint accuracy | First CE <= 0.8 (per seed) |
|---|---:|---:|---|
| mlp | 0.002013 | 100.000% | [6, 7, 6] |
| qnn_random | 0.013360 | 100.000% | [22, 17, 19] |
| qnn_small | 0.013650 | 100.000% | [12, 14, 14] |
| qnn_small_frozen | 0.059744 | 100.000% | [20, 21, 22] |

## Physical error evaluation

All rates below include the disclosed ideal final cleanup. See evaluation.json for raw recovery failures, detectable residuals and Wilson intervals.

| Model | p | q | Logical error, mean across seeds |
|---|---:|---:|---:|
| lookup | 0.01 | 0.0 | 0.013672 |
| lookup | 0.01 | 0.03 | 0.028809 |
| lookup | 0.01 | 0.08 | 0.055176 |
| lookup | 0.05 | 0.0 | 0.196777 |
| lookup | 0.05 | 0.03 | 0.239746 |
| lookup | 0.05 | 0.08 | 0.289307 |
| lookup | 0.1 | 0.0 | 0.442871 |
| lookup | 0.1 | 0.03 | 0.470215 |
| lookup | 0.1 | 0.08 | 0.507080 |
| mlp | 0.01 | 0.0 | 0.013672 |
| mlp | 0.01 | 0.03 | 0.028809 |
| mlp | 0.01 | 0.08 | 0.055176 |
| mlp | 0.05 | 0.0 | 0.196777 |
| mlp | 0.05 | 0.03 | 0.239746 |
| mlp | 0.05 | 0.08 | 0.289307 |
| mlp | 0.1 | 0.0 | 0.442871 |
| mlp | 0.1 | 0.03 | 0.470215 |
| mlp | 0.1 | 0.08 | 0.507080 |
| gru | 0.01 | 0.0 | 0.013346 |
| gru | 0.01 | 0.03 | 0.026367 |
| gru | 0.01 | 0.08 | 0.050130 |
| gru | 0.05 | 0.0 | 0.192708 |
| gru | 0.05 | 0.03 | 0.229818 |
| gru | 0.05 | 0.08 | 0.276449 |
| gru | 0.1 | 0.0 | 0.435384 |
| gru | 0.1 | 0.03 | 0.465007 |
| gru | 0.1 | 0.08 | 0.501465 |
| qnn_random | 0.01 | 0.0 | 0.013672 |
| qnn_random | 0.01 | 0.03 | 0.028809 |
| qnn_random | 0.01 | 0.08 | 0.055176 |
| qnn_random | 0.05 | 0.0 | 0.196777 |
| qnn_random | 0.05 | 0.03 | 0.239746 |
| qnn_random | 0.05 | 0.08 | 0.289307 |
| qnn_random | 0.1 | 0.0 | 0.442871 |
| qnn_random | 0.1 | 0.03 | 0.470215 |
| qnn_random | 0.1 | 0.08 | 0.507080 |
| qnn_small | 0.01 | 0.0 | 0.013672 |
| qnn_small | 0.01 | 0.03 | 0.028809 |
| qnn_small | 0.01 | 0.08 | 0.055176 |
| qnn_small | 0.05 | 0.0 | 0.196777 |
| qnn_small | 0.05 | 0.03 | 0.239746 |
| qnn_small | 0.05 | 0.08 | 0.289307 |
| qnn_small | 0.1 | 0.0 | 0.442871 |
| qnn_small | 0.1 | 0.03 | 0.470215 |
| qnn_small | 0.1 | 0.08 | 0.507080 |
| qnn_small_frozen | 0.01 | 0.0 | 0.013672 |
| qnn_small_frozen | 0.01 | 0.03 | 0.028809 |
| qnn_small_frozen | 0.01 | 0.08 | 0.055176 |
| qnn_small_frozen | 0.05 | 0.0 | 0.196777 |
| qnn_small_frozen | 0.05 | 0.03 | 0.239746 |
| qnn_small_frozen | 0.05 | 0.08 | 0.289307 |
| qnn_small_frozen | 0.1 | 0.0 | 0.442871 |
| qnn_small_frozen | 0.1 | 0.03 | 0.470215 |
| qnn_small_frozen | 0.1 | 0.08 | 0.507080 |

## Gradient scaling

Slopes are descriptive log10(variance) per additional qubit; finite-range fits do not prove an asymptotic law.

| Depth | Initialization | Cost | Slope |
|---|---|---|---:|
| 1 | random | global | -0.47490 |
| 1 | random | local | -0.16234 |
| 1 | small | global | -0.13229 |
| 1 | small | local | -0.17295 |
| 3 | random | global | -0.48407 |
| 3 | random | local | -0.17074 |
| 3 | small | global | -0.12313 |
| 3 | small | local | -0.16025 |
| 6 | random | global | -0.52430 |
| 6 | random | local | -0.18477 |
| 6 | small | global | -0.12661 |
| 6 | small | local | -0.15627 |

At depth 6, global-cost slopes are -0.524 for random initialization and -0.127 for small-angle initialization. Random/local gives -0.185. Both strategies slow the measured finite-range decay.

## Important limits

- GRU receives history; other decoders receive the latest round.
- Table task trains and evaluates on all 64 syndromes; this is not held-out generalization.
- Small-angle versus random initialization is a same-architecture comparison.
- Frozen quantum features control for readout-only learning.
- Three seeds and a finite qubit range do not establish universal advantages.
- No quantum hardware, finite shots or local gate-noise experiment is included.