# LyricPredict

LyricPredict is a local Web app for Chinese/English lyric continuation. It lets a user import `.txt` or `.lrc` lyric files, enter a short context, and predict one likely next lyric phrase ending at `,`, `.`, `，`, or `。`. If the confidence gate rejects the result, the app returns no text.

## Setup

```powershell
conda env create -f environment.yml
conda activate lyricpredict
```

If you prefer pip inside an existing environment:

```powershell
pip install -r requirements.txt
```

The default model is `uer/gpt2-distil-chinese-cluecorpussmall`, which is small enough for CPU inference after the first download.

## Prepare Lyrics

Put raw `.txt` and `.lrc` files in `data/raw`, then run:

```powershell
python -m lyricpredict.prepare --config configs/default.yaml
```

The app can also import files from the browser and will clean them into `data/processed`.

## Train

```powershell
python -m lyricpredict.train --config configs/default.yaml
```

Training uses LoRA when `peft` is installed and writes output under `models/default`.

## Calibrate Confidence

```powershell
python -m lyricpredict.calibrate --config configs/default.yaml
```

This writes `models/default/confidence.json`. Predictions below the calibrated threshold return an empty string.

## Serve

```powershell
python -m lyricpredict.serve --config configs/default.yaml
```

Open `http://127.0.0.1:8000`. Use the continue button or `F8` to generate the next sentence.
