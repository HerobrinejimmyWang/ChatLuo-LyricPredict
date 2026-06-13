# Matching Model Evaluation

## closed-library

### Recall (N=2009)

Cell format: correct / abstain / wrong.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Multi-sentences (693) | 442 / 249 / 2 | 682 / 9 / 2 | 684 / 7 / 2 |
| Single sentence (658) | 652 / 6 / 0 | 658 / 0 / 0 | 658 / 0 / 0 |
| Complex context (658) | 648 / 10 / 0 | 656 / 2 / 0 | 656 / 2 / 0 |

### Recall Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Multi-sentences | 179.71 / 174.11 ms | 181.83 / 174.55 ms | 185.10 / 180.10 ms |
| Single sentence | 45.27 / 42.82 ms | 45.27 / 43.38 ms | 44.63 / 41.62 ms |
| Complex context | 158.71 / 158.39 ms | 163.41 / 159.79 ms | 161.73 / 158.98 ms |
