# Final Matching Model Evaluation

This report summarizes the current final closed-library `char-match` backend evaluation after metadata cleaning.

Cell format:

`correct / abstain / wrong`

Latency was measured on CPU: 13th Gen Intel(R) Core(TM) i9-13900H.

## Dataset

The processed dataset was rebuilt from:

- `selfdata/selflyricdata`
- `selfdata/selflyricdata2`

Prepared stats:

- files: 171
- songs: 152
- cleaned lyric lines: 8218
- train lines: 7396
- valid lines: 822

Metadata colon check inside `data/processed/songs.jsonl` line payloads:

- Chinese colon `：`: 0
- ASCII colon `:`: 0

## Recall

Sample size: `N=120`, 40 cases per scenario.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Multi-sentences (40) | 23 / 17 / 0 | 39 / 1 / 0 | 39 / 1 / 0 |
| Single sentence (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |
| Complex context (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |

### Recall Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Multi-sentences | 147.50 / 144.89 ms | 163.45 / 158.11 ms | 143.61 / 139.78 ms |
| Single sentence | 37.41 / 33.76 ms | 39.00 / 38.94 ms | 39.51 / 37.87 ms |
| Complex context | 141.17 / 139.48 ms | 139.88 / 140.93 ms | 134.11 / 132.82 ms |

## Situations

Sample size: `N=60`, 10 cases per scenario.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Half-sentences (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Symbols Outputs (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-one (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-two (10) | 9 / 1 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Mixed Context (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Out-of-library (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |

### Correction Full

Requires correct output and correct `corrected_context`.

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Correction-one | 10/10 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) |
| Correction-two | 9/10 (90.0%) | 10/10 (100.0%) | 10/10 (100.0%) |

### Situation Latency mean / median

| Scenario | char-match:strict | char-match:balanced | char-match:tolerant |
|----------|-------------------|---------------------|---------------------|
| Half-sentences | 10.90 / 9.85 ms | 10.12 / 9.97 ms | 10.88 / 10.36 ms |
| Symbols Outputs | 20.78 / 20.11 ms | 20.90 / 21.00 ms | 20.43 / 20.38 ms |
| Correction-one | 30.33 / 27.38 ms | 31.23 / 27.69 ms | 31.50 / 27.95 ms |
| Correction-two | 34.16 / 37.42 ms | 34.28 / 37.79 ms | 35.12 / 39.62 ms |
| Mixed Context | 25.10 / 26.49 ms | 25.03 / 26.50 ms | 24.95 / 25.87 ms |
| Out-of-library | 65.27 / 67.14 ms | 65.34 / 66.67 ms | 65.62 / 67.33 ms |

## Conclusion

The current recommended route remains `char-match:balanced`.

After metadata cleaning, balanced has no wrong outputs in the sampled Recall or Situation evaluations. The remaining Recall issue is one abstain in Multi-sentences. Situation and Correction Full are both clean for balanced and tolerant in this sample.
