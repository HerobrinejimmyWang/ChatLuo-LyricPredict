# Matching Model Evaluation

## closed-library

### Recall (N=120)

Cell format: correct / abstain / wrong.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Multi-sentences (40) | 20 / 20 / 0 | 37 / 3 / 0 | 38 / 2 / 0 |
| Single sentence (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |
| Complex context (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |

### Recall Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Multi-sentences | 77.89 / 76.47 ms | 83.95 / 79.98 ms | 101.10 / 94.06 ms |
| Single sentence | 15.70 / 15.88 ms | 16.26 / 16.59 ms | 15.69 / 16.06 ms |
| Complex context | 77.52 / 76.25 ms | 75.66 / 74.75 ms | 74.28 / 75.09 ms |
