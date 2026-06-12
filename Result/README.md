# LyricPredict Result

本目录保留当前后端匹配式 model 实验的最终版结果，以及最初设计测试用例时使用的说明文件。

## Final Results

- `final/final_matching_evaluation.md`：最终汇总表，覆盖 Recall、Situation、负例拒答、Correction Full 和延迟。
- `final/recall.sample120.md`：Recall 原始评估输出，样本数 120。
- `final/situations_with_negative.sample60.md`：Situation 原始评估输出，样本数 60，包含 `Mixed Context` 和 `Out-of-library`。
- `final/failures.recall.sample120.json`：Recall 失败样例清单。
- `final/failures.situations_with_negative.sample60.json`：Situation 失败样例清单。

## Original Test Design

- `design/testcase.md`
- `design/testresult.md`
- `design/testresult.generated.md`

## Selected Result Snapshot

当前建议优先观察 `char-match:balanced`：

- Recall: `Multi-sentences 37 / 3 / 0`，`Single sentence 40 / 0 / 0`，`Complex context 40 / 0 / 0`。
- Situation: `Half-sentences`、`Symbols Outputs`、`Correction-one`、`Mixed Context`、`Out-of-library` 均无错误输出。
- `Correction-two` 当前为 `8 / 2 / 0`，主要问题是拒答，不是错误输出。

单元与回归测试通过：`125 passed`。
