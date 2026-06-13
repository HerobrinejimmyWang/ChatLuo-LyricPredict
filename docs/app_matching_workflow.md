# LyricPredict App Matching Workflow

## Current Default Route

The app now uses the closed-library `matching` route by default:

```text
input context
  -> extract lyric-like context
  -> load data/processed/matching_index.json
  -> char-match ranker
  -> strictness gate
  -> output / abstain / corrected_context
```

The app no longer exposes separate `retrieval` and `model-only` choices. The old Transformer/LoRA and legacy n-gram code paths remain available for backend experiments and benchmarks, but they are not the default app workflow.

## Runtime Files

Runtime prediction depends on the cleaned data cache, not on a separately trained neural weight file:

- `data/processed/songs.jsonl`: cleaned lyric lines grouped by source song.
- `data/processed/matching_index.json`: cached candidate library used by `char-match`.
- `data/processed/source_manifest.json`: source directories and cleaning stats.
- `data/processed/near_duplicate_report.json`: exact and near-duplicate song report for human review.
- `data/processed/stats.json`: file/song/line counts.

If `matching_index.json` is missing or stale, the backend rebuilds it from `songs.jsonl`.

## One-Click Build

The desktop app's update button now runs the backend preparation path only:

```powershell
python -m lyricpredict.training_pipeline `
  --model-id default `
  --skip-transformer `
  --skip-ngram `
  --skip-calibrate
```

For the default profile, the registered source directories are:

```text
selfdata/selflyricdata
selfdata/selflyricdata2
```

Adding a new dataset to the same profile appends it to `data_dirs`; rebuilding regenerates the cleaned cache and matching index.

## User-Facing Controls

The app still keeps:

- `strict`: higher precision, more abstain.
- `balanced`: recommended default.
- `tolerant`: more recall, still gated against mixed/out-of-library inputs.
- correction toggle: returns `corrected_context` when a near-match suggests that the user's previous context contains a small typo.

The backend reason strings are mapped to short user-facing labels such as `matched`, `partial matched`, `low confidence`, `ambiguous match`, and `no match`.

## Backend Boundaries

- `char-match` is the selected app route.
- `legacy-ngram-generator` is retained only as a benchmark/compatibility mechanism.
- Transformer/LoRA remains an experiment path and is not called by the app default route.
- Near-duplicate songs are reported, not automatically merged; later app UX can offer "keep / remove / review" decisions.
