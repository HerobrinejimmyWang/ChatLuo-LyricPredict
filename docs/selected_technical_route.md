# Selected Technical Route

## Decision

当前后端 model 实验优先采用 closed-library matching 路线，核心实现为 `char-match`。它不做自由生成，所有输出都来自已清洗歌词候选库中的真实片段。

暂不把 Transformer / LoRA 作为默认 CPU 端 model 路线。Qwen2.5-0.5B LoRA 虽然比 GPT-2 0.1B 更有语言能力，但 CPU 推理延迟过高；GPT-2 0.1B 的生成质量不足。legacy n-gram generator 保留为 benchmark 和后端兼容机制，不作为新整体流程的直接输出链路。

## Model Boundary

`model` 位置先定义为匹配式后端：

```text
context -> candidate library -> char-match ranker -> confidence gate -> output / abstain
```

候选库来自 `data/processed/songs.jsonl`，构造内容包括：

- 下一句片段：用于 Recall。
- 半句候选：用于 Half-sentences。
- 带分隔符语义的变体：用于 Symbols Outputs。
- 标准上下文：用于纠错时返回 `corrected_context`。

## Scoring Strategy

`char-match` 采用多级匹配：

- 精确/后缀匹配：当前上下文末尾命中训练歌词时高置信输出。
- 前缀/半句匹配：用于句内续写，保证半句场景不额外补分隔符。
- 字符 n-gram overlap：处理轻微错字和变体输入。
- 连续覆盖率与嵌入上下文降权：防止输入中较早位置包含某句歌词，但末尾已经混入其它歌词时误输出。

三档严格度只调整是否输出，不改变候选内容：

- `strict`：更偏拒答。
- `balanced`：当前推荐默认。
- `tolerant`：更偏召回，但仍要求负例拒答。

## Evaluation Criteria

报告统一使用：

```text
正确输出 / 拒绝输出 / 错误输出
```

错误输出比拒绝输出更严重，因此负例场景必须优先保证错误输出为 0：

- `Mixed Context`：句内混入来自不同歌词的片段，应拒答。
- `Out-of-library`：不属于训练库的普通文本，应拒答。

纠错场景需要额外看 `Correction Full`：只有预测输出正确且 `corrected_context` 也正确，才算完整满足 app 端纠错语义。

## Current Result

最终结果见 `Result/final/final_matching_evaluation.md`。

推荐路线：`char-match:balanced`。

当前主要结论：

- Single sentence 与 Complex context 抽样召回满分。
- Multi-sentences 仍有少量拒答，但没有错误输出。
- Half-sentences、Symbols Outputs、Correction-one、Mixed Context、Out-of-library 表现稳定。
- Correction-two 的主要缺口是拒答，可后续通过更强的纠错候选生成或轻量 ranker 优化。

## Next Integration Step

接入 app workflow 时建议保持后端边界清晰：

- `char-match` 作为 model-only 的首选候选。
- auto 链路可先走严格 retriever；retriever 不过时调用 `char-match:balanced`。
- legacy n-gram generator 只保留为 benchmark 或隐藏 fallback，不直接进入新的整体流程。
- 前端无需暴露复杂后端逻辑，只展示输出、拒答、纠错提示和可读 reason。
