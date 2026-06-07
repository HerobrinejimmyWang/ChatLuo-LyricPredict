# LyricPredict

LyricPredict is a local, personal-use lyric continuation project. It imports `.txt` and `.lrc` lyric files, fine-tunes or loads a small Transformer causal language model, and predicts one following lyric sentence at a time.

Core context:

- CPU inference must work.
- Generation stops at an English or Chinese comma/period.
- Low-confidence, garbled, repetitive, or unfinished outputs must return an empty string.
- The Web UI supports lyric upload, context input, a continue button, and `F8` for continuing the next sentence.
- Main configuration lives in `configs/default.yaml`; the default fine-tuned output is usually `models/default`.

Useful commands:

```powershell
python -m lyricpredict.prepare --config configs/default.yaml
python -m lyricpredict.train --config configs/default.yaml
python -m lyricpredict.calibrate --config configs/default.yaml
python -m lyricpredict.serve --config configs/default.yaml --host 127.0.0.1 --port 8002
python -m pytest -q
```

Local data and model artifacts include `selflyricdata`, `data/processed`, and `models`. Avoid mixing testcase lyrics into training data when changing training, evaluation, or filtering rules.
