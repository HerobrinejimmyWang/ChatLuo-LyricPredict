# LyricPredict Result

This directory keeps the final backend matching-model evaluation outputs and the original test-design files.

## Final Results

- `final/final_matching_evaluation.md`: final summary table after metadata cleaning.
- `final/recall.sample120.md`: raw Recall evaluation output, sample size 120.
- `final/situations_with_negative.sample60.md`: raw Situation evaluation output, sample size 60, including `Mixed Context` and `Out-of-library`.
- `final/failures.recall.sample120.json`: Recall failure examples.
- `final/failures.situations_with_negative.sample60.json`: Situation failure examples.

## Expanded Validation

- `expanded/current_validation_summary.md`: current dataset 10 percent Recall and expanded Situation summary.
- `expanded/current_recall.ratio10.md`: current dataset Recall at about 10 percent of full evaluable cases.
- `expanded/current_situations.expanded.md`: current dataset Situation validation with doubled Half/Correction/Mixed samples.
- `xiong/final_matching_evaluation.md`: validation on `selfdata/Lyrics_xiong`.
- `xiong/recall.sample120.md`: xiong Recall evaluation.
- `xiong/situations_with_negative.sample60.md`: xiong Situation evaluation.

## Original Test Design

- `design/testcase.md`
- `design/testresult.md`
- `design/testresult.generated.md`

## Selected Result Snapshot

Recommended column: `char-match:balanced`.

- Recall: `Multi-sentences 39 / 1 / 0`, `Single sentence 40 / 0 / 0`, `Complex context 40 / 0 / 0`.
- Situations: all six scenarios are `10 / 0 / 0` under balanced.
- Correction Full: `Correction-one 10/10`, `Correction-two 10/10` under balanced.
- Metadata colon check inside processed lyric lines: `： = 0`, `: = 0`.
