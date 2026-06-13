# Matching Model Evaluation

## closed-library

### Recall (N=120)

Cell format: correct / abstain / wrong.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Multi-sentences (40) | 30 / 10 / 0 | 39 / 1 / 0 | 40 / 0 / 0 |
| Single sentence (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |
| Complex context (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |

### Recall Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Multi-sentences | 199.21 / 192.22 ms | 197.36 / 190.11 ms | 197.09 / 188.98 ms |
| Single sentence | 53.60 / 54.25 ms | 54.16 / 51.79 ms | 57.37 / 56.98 ms |
| Complex context | 178.78 / 180.22 ms | 177.77 / 183.33 ms | 171.51 / 170.23 ms |
