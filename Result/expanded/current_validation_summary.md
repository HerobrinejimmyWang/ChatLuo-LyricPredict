# Current Dataset Expanded Validation

Dataset:

- source dirs: `selfdata/selflyricdata`, `selfdata/selflyricdata2`
- prepared stats: 171 files, 152 songs, 8220 cleaned lines

## Recall 10 Percent

Command output: `Result/expanded/current_recall.ratio10.md`

Failure examples: `Result/expanded/failures.current_recall.ratio10.json`

Sample size: `N=2009`.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Multi-sentences (693) | 442 / 249 / 2 | 682 / 9 / 2 | 684 / 7 / 2 |
| Single sentence (658) | 652 / 6 / 0 | 658 / 0 / 0 | 658 / 0 / 0 |
| Complex context (658) | 648 / 10 / 0 | 656 / 2 / 0 | 656 / 2 / 0 |

Notes:

- `char-match:balanced` remains strong at larger sample size.
- The two balanced wrong outputs are exact-match formatting misses: same lyric content, but internal spaces differ from expected text.
- Multi-sentence abstains are mostly ambiguity gates.

## Expanded Situations

Command output: `Result/expanded/current_situations.expanded.md`

Failure examples: `Result/expanded/failures.current_situations.expanded.json`

Sample size: `N=100`.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Half-sentences (20) | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| Symbols Outputs (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-one (20) | 18 / 2 / 0 | 19 / 1 / 0 | 19 / 1 / 0 |
| Correction-two (20) | 11 / 9 / 0 | 18 / 2 / 0 | 18 / 2 / 0 |
| Mixed Context (20) | 19 / 0 / 1 | 19 / 0 / 1 | 19 / 0 / 1 |
| Out-of-library (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |

Notes:

- Half-sentences and out-of-library rejection stay stable.
- Correction failures are abstains, not wrong outputs.
- The one Mixed Context wrong output is caused by two near-duplicate versions of the same song (`东京不太热`), so the negative-case generator should later avoid near-duplicate sources.
