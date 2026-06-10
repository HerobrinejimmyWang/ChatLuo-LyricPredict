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

The default model is `souljoy/gpt2-small-chinese-cluecorpussmall`, which provides safetensors weights on the main branch and can run on CPU after the first download. You can switch to another Chinese causal language model in `configs/default.yaml` as long as it can be loaded safely by your local PyTorch/Transformers versions.

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

For a repeatable backend training flow, prefer the pipeline entrypoint. It keeps
model paths and accumulated data directories in `configs/models.yaml`, writes a
runtime config under `configs/models/`, and records the last run in
`models/<model-id>/training_pipeline.json`.

Create a new model from one dataset:

```powershell
python -m lyricpredict.training_pipeline --model-name LuoLocal --data-dir selfdata/selflyricdata
```

Append another dataset to the same model later:

```powershell
python -m lyricpredict.training_pipeline --model-id luolocal --data-dir selfdata/selflyricdata2
```

The second command reuses the model profile and prepares all accumulated data
directories. Use `--replace-data` only when you intentionally want to discard
the previous source list.

For a quick CPU-friendly n-gram/model-only rebuild without LoRA fine-tuning:

```powershell
python -m lyricpredict.training_pipeline --model-id luolocal --data-dir selfdata/selflyricdata2 --skip-transformer
```

To preview the exact commands without running them:

```powershell
python -m lyricpredict.training_pipeline --model-id luolocal --data-dir selfdata/selflyricdata2 --dry-run
```

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

## Windows Desktop App

The optional Windows shell runs as a tray app with a settings window and a suggestion line. Install the desktop-only dependencies first:

```powershell
pip install PySide6 pywin32 pywinauto pyinstaller
```

Then start it with:

```powershell
python -m lyricpredict.desktop --settings configs/app.yaml
```

Use `Ctrl+Alt+L` to read the current selection, `Tab` to accept a suggestion, and `Esc` to reject it. The desktop app reuses `configs/default.yaml` for LyricPredict inference settings and stores desktop preferences in `configs/app.yaml`.

Auto read can be enabled from the settings window. It uses Windows UI Automation to read the focused text control after enough typed changes and never uses the clipboard in automatic mode.
In `used-windows` scope, pressing the trigger hotkey once in a target window records that window type in `configs/app_state.yaml` so future app launches can allow auto read for the same app/control class.

Suggestion styles live under `assets/suggestion_styles`; see `docs/suggestion_style_process.md` for the LUO fused-box workflow and pointer anchoring rules.
