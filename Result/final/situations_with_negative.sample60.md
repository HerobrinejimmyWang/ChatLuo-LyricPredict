# Matching Model Evaluation

## closed-library

### Situations (N=60)

Cell format: correct / abstain / wrong.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Half-sentences (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Symbols Outputs (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-one (10) | 9 / 1 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-two (10) | 6 / 4 / 0 | 8 / 2 / 0 | 8 / 2 / 0 |
| Mixed Context (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Out-of-library (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |

### Correction Full

Requires correct output and correct `corrected_context`.

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Correction-one | 9/10 (90.0%) | 10/10 (100.0%) | 10/10 (100.0%) |
| Correction-two | 6/10 (60.0%) | 8/10 (80.0%) | 8/10 (80.0%) |

### Situation Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Half-sentences | 11.25 / 11.91 ms | 10.59 / 11.09 ms | 11.39 / 11.20 ms |
| Symbols Outputs | 18.58 / 16.05 ms | 18.68 / 15.57 ms | 17.03 / 15.55 ms |
| Correction-one | 26.30 / 26.80 ms | 23.20 / 25.69 ms | 23.52 / 23.38 ms |
| Correction-two | 27.73 / 25.80 ms | 30.27 / 27.85 ms | 29.60 / 27.36 ms |
| Mixed Context | 25.92 / 26.19 ms | 25.34 / 25.47 ms | 24.68 / 25.84 ms |
| Out-of-library | 65.51 / 65.12 ms | 69.66 / 70.64 ms | 66.89 / 68.49 ms |
