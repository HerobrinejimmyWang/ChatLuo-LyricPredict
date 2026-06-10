# Results

## Dataset

`selfdata\selflyricdata + selfdata\selflyricdata2`

## Recall

取每行歌词的作为输入，下一行为预期输出，评估模型正确输出的召回率（因为所有歌词数据都在训练数据中，用召回来表征）。

- Multi-sentences: 在提取输入歌词时，向前延伸一定长度，使之达到 32 tokens 的上下文长度
- Single sentence: 仅包含提取一句的歌词
- Complex context: 在提取的歌词前混入一段无关文本

- Multi-sentences: enough length for 32 tokens context window
- Single sentence: only one sentence before predict
- Complex context: contain a non-relative sentence in the context window

*在测试前需清除单句歧义句，即出现多次无法定位上下文的句子*
*Needed to remove the conflits in 'Single sentence' scene*

|                 | Transformer | Model-only | Auto |
|-----------------|-------------|------------|------|
| Multi-sentences | SKIPPED     | 0/20 (0.0%) | 17/20 (85.0%) |
| Single sentence | SKIPPED     | 0/20 (0.0%) | 18/20 (90.0%) |
| Complex context | SKIPPED     | 0/20 (0.0%) | 0/20 (0.0%) |


## ACC for Some Situations

此部分从数据中抽样约 100 句歌词设计为测试样本测试即可。

- Half-sentences：当输入在句中停止时，要求正确输出后半句，无多余标点
- Symbols Outputs：正确处理不同的标点符号，包括用户已输入 `，`` ``。` 的情况，无多余输出；缺少标点时按默认选择补齐
- Correction-one：当输入有单字错误时（一般语义接近或读音接近），模型能够正确识别并给予纠正的能力
- Correction-two：当输入有两字错误时（一般语义接近或读音接近），模型能够正确识别并给予纠正的能力

|                 | Transformer | Model-only | Auto |
|-----------------|-------------|------------|------|
| Half-sentences  | SKIPPED     | 0/50 (0.0%) | 31/50 (62.0%) |
| Symbols Outputs | SKIPPED     | 1/50 (2.0%) | 39/50 (78.0%) |
| Correction-one  | SKIPPED     | 2/50 (4.0%) | 48/50 (96.0%) |
| Correction-two  | SKIPPED     | 2/50 (4.0%) | 41/50 (82.0%) |

Note: this run uses the n-gram backup model `models/full_2datasets_20260610` trained from `selfdata/selflyricdata + selfdata/selflyricdata2`. `Transformer` was skipped. Recall rows use fixed-seed samples of 20 cases per scene; situation rows use fixed-seed samples of 50 cases per scene.
