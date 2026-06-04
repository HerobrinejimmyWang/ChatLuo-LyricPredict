# LyricPredict Agent Notes

LyricPredict is a local personal-use lyric continuation app. It imports `.lrc`/`.txt` lyrics, fine-tunes a small CPU-capable Chinese GPT-2 model with LoRA, and predicts one following lyric sentence up to `,`, `.`, `，`, or `。`. Low-confidence or malformed generations must return no text.

## Environment

- Conda env: `lyricpredict`
- Python entrypoints:
  - `python -m lyricpredict.prepare --config configs/default.yaml`
  - `python -m lyricpredict.train --config configs/default.yaml`
  - `python -m lyricpredict.calibrate --config configs/default.yaml`
  - `python -m lyricpredict.serve --config configs/default.yaml --host 127.0.0.1 --port 8002`

## Current Model/Data

- Base model: `souljoy/gpt2-small-chinese-cluecorpussmall`
- Fine-tuned adapter output: `models/default`
- User lyric source: `selflyricdata`
- Processed data: `data/processed`
- Training filters skip non-Chinese or high-`[UNK]` lines; line endings are normalized by appending `。` during training when no terminator exists.

## Implementation Notes

- Generation uses multiple sampling attempts (`generation_attempts`) and rejects bad outputs via confidence gates.
- Token decoding must decode the whole generated sequence, not token-by-token, to avoid WordPiece `##` artifacts.
- Keep output empty when rejected; the Web UI should not append rejected text.
- The model may generate Simplified/Traditional mixed Chinese. Prefer normalization/post-processing over assuming embeddings are aligned.

## Verification

Run:

```powershell
$env:TMP = (Join-Path (Get-Location) '.tmp')
$env:TEMP = $env:TMP
python -m pytest -q --basetemp .tmp\pytest -o cache_dir=.tmp\pytest-cache
```

