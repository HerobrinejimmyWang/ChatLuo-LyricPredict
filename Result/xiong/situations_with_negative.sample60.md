# Matching Model Evaluation

## closed-library

### Situations (N=60)

Cell format: correct / abstain / wrong.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Half-sentences (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Symbols Outputs (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-one (10) | 8 / 2 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-two (10) | 6 / 4 / 0 | 8 / 2 / 0 | 9 / 1 / 0 |
| Mixed Context (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Out-of-library (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |

### Correction Full

Requires correct output and correct `corrected_context`.

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Correction-one | 8/10 (80.0%) | 10/10 (100.0%) | 10/10 (100.0%) |
| Correction-two | 6/10 (60.0%) | 8/10 (80.0%) | 9/10 (90.0%) |

### Situation Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Half-sentences | 92.30 / 37.15 ms | 34.91 / 35.38 ms | 32.26 / 33.80 ms |
| Symbols Outputs | 64.41 / 43.30 ms | 64.64 / 44.42 ms | 63.51 / 44.34 ms |
| Correction-one | 62.34 / 57.74 ms | 69.09 / 68.13 ms | 63.39 / 60.33 ms |
| Correction-two | 66.84 / 62.57 ms | 69.87 / 65.96 ms | 69.65 / 68.88 ms |
| Mixed Context | 73.66 / 66.23 ms | 70.43 / 67.56 ms | 70.64 / 68.56 ms |
| Out-of-library | 152.50 / 149.96 ms | 151.30 / 153.05 ms | 145.92 / 145.00 ms |
