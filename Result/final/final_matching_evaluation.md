# Final Matching Model Evaluation

本报告汇总当前最终采用的 closed-library `char-match` 后端评估结果。单元格格式为：

`正确输出 / 拒绝输出 / 错误输出`

## Recall

样本数：`N=120`，每个场景 40 条。

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

## Situations

样本数：`N=60`，每个场景 10 条。

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Half-sentences (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Symbols Outputs (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-one (10) | 9 / 1 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-two (10) | 6 / 4 / 0 | 8 / 2 / 0 | 8 / 2 / 0 |
| Mixed Context (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Out-of-library (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |

### Correction Full

要求输出正确，且 `corrected_context` 同时正确。

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

## Conclusion

当前推荐将 `char-match:balanced` 作为后续接入 workflow 的候选后端：它在主要 Recall 场景上保持高正确率，在负例场景中没有错误输出，剩余主要问题集中在二字纠错时的拒答。
