# Matching Model Evaluation

## closed-library

### Situations (N=100)

Cell format: correct / abstain / wrong.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Half-sentences (20) | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| Symbols Outputs (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-one (20) | 18 / 2 / 0 | 19 / 1 / 0 | 19 / 1 / 0 |
| Correction-two (20) | 11 / 9 / 0 | 18 / 2 / 0 | 18 / 2 / 0 |
| Mixed Context (20) | 19 / 0 / 1 | 19 / 0 / 1 | 19 / 0 / 1 |
| Out-of-library (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |

### Correction Full

Requires correct output and correct `corrected_context`.

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Correction-one | 18/20 (90.0%) | 19/20 (95.0%) | 19/20 (95.0%) |
| Correction-two | 11/20 (55.0%) | 18/20 (90.0%) | 18/20 (90.0%) |

### Situation Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Half-sentences | 32.50 / 30.92 ms | 32.42 / 33.26 ms | 29.60 / 30.86 ms |
| Symbols Outputs | 57.12 / 53.91 ms | 58.63 / 53.49 ms | 60.04 / 57.48 ms |
| Correction-one | 67.42 / 54.15 ms | 68.07 / 58.84 ms | 66.34 / 56.27 ms |
| Correction-two | 75.04 / 71.37 ms | 74.38 / 69.76 ms | 75.62 / 70.73 ms |
| Mixed Context | 56.17 / 53.78 ms | 55.24 / 53.70 ms | 56.36 / 55.58 ms |
| Out-of-library | 130.60 / 128.67 ms | 132.57 / 134.96 ms | 133.45 / 132.31 ms |
