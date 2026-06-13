# Lyrics Xiong Matching Evaluation

This report validates the same `char-match` closed-library flow on another lyric dataset.

Dataset:

- source dir: `selfdata/Lyrics_xiong`
- config: `configs/models/xiong_char_match.yaml`
- prepared stats: 320 files, 315 songs, 19501 cleaned lines

Metadata check after cleaning:

- obvious production metadata with colon was removed
- remaining colon lines are mostly lyric content, time expressions, or dialogue inside lyric text

Cell format:

`correct / abstain / wrong`

## Recall

Raw report: `Result/xiong/recall.sample120.md`

Failure examples: `Result/xiong/failures.recall.sample120.json`

Sample size: `N=120`.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Multi-sentences (40) | 30 / 10 / 0 | 39 / 1 / 0 | 40 / 0 / 0 |
| Single sentence (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |
| Complex context (40) | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |

## Situations

Raw report: `Result/xiong/situations_with_negative.sample60.md`

Failure examples: `Result/xiong/failures.situations_with_negative.sample60.json`

Sample size: `N=60`.

| Scenario (N) | char-match:strict | char-match:balanced | char-match:tolerant |
|--------------|-------------------|---------------------|---------------------|
| Half-sentences (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Symbols Outputs (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-one (10) | 8 / 2 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Correction-two (10) | 6 / 4 / 0 | 8 / 2 / 0 | 9 / 1 / 0 |
| Mixed Context (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| Out-of-library (10) | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |

## Conclusion

The flow transfers well to `Lyrics_xiong`.

- `char-match:balanced` has no wrong outputs in the sampled recall and situation evaluations.
- Recall is nearly full: only one Multi-sentences abstain.
- Situation robustness is good; the remaining weakness is Correction-two abstain.
- `char-match:tolerant` recovers more xiong recall and Correction-two cases while still keeping wrong outputs at zero in this sample.
