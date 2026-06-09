# Suggestion Style Process

This process turns a visual reference into a LyricPredict desktop suggestion style.

## Asset Layout

Create one folder per style:

```text
assets/suggestion_styles/<style_id>/
  reference.png
  character.png
  box.png
  style.yaml
```

- `reference.png`: the original design reference.
- `character.png`: transparent high-resolution character layer used by the desktop APP.
- `box.png`: transparent fused suggestion box used as the default LUO visual surface.
- `style.yaml`: stable metadata for the crop, character size, bubble colors, fused canvas layout, and pointer anchoring.

Use ASCII file names inside `assets/` so packaging and Windows console encoding stay predictable.

## Build Assets

Run the helper script from the project root:

```powershell
python scripts/make_suggestion_style.py `
  --source .design\Style1-LUO.png `
  --out-dir assets\suggestion_styles\luo `
  --crop 0 0 760 430
```

The script copies the source to `reference.png` and creates `character.png` by:

- cropping the character region,
- converting near-white background pixels to transparent pixels,
- preserving the character pose and outline.

Adjust `--crop X Y W H` if the character shifts in a future design.

Then build the fused LUO box:

```powershell
python scripts/build_luo_box.py
```

This creates `box.png`, a wider regular box without the speech-tail arrow. The desktop APP draws this full asset first, then overlays suggestion text inside the reserved content area.

For LUO-style boxes, the APP positions the popup by the visible bubble rectangle, not the transparent window bounds. This matters because the character can extend above or outside the actual text box.

## Style Metadata

Use `style.yaml` to document how the APP should interpret the assets:

```yaml
name: Style Name
reference: reference.png
box: box.png
character:
  source: character.png
  crop: [0, 0, 760, 430]
  width: 285
  anchor: top-left
  position: [26, 0]
bubble:
  background: "#fffefe"
  border: "#7d6bd6"
  border_width: 2
  radius: 18
  text: "#2a2540"
  meta: "#6f66a8"
  rect: [34, 148, -68, 158]
  text_rect: [320, 0.58h, -382, -62]
  meta_rect: [320, -52, -382, 24]
nine_slice:
  source_size: [1536, 1024]
  box_size: [960, 340]
  protected_character: [0, 0, 760, 430]
  stretch_center: [170, 430, 1260, 420]
window:
  default_width: 720
  pointer_gap: 34
  margin: 8
```

`bubble.rect` is the visible bubble area in `box.png` coordinates. Negative values for right/bottom mean "subtract from the full width/height" where used in documentation, while the current LUO runtime uses the concrete generated dimensions `960x340`.

Pointer positioning uses the visible bubble:

- `top-left`: bubble bottom-right stays above-left of the cursor by `pointer_gap`.
- `bottom-right`: bubble top-left stays below-right of the cursor by `pointer_gap`.
- The final popup is clamped inside the current monitor with `margin`.

## APP Integration

For a new style:

1. Add the style id to `VALID_SUGGESTION_STYLES` in `lyricpredict/desktop/settings.py`.
2. Add it to the suggestion style combo in `lyricpredict/desktop/qt_app.py`.
3. Add a style branch in `SuggestionLine._apply_style()`.
4. Add or reuse a fused canvas widget that paints the character and bubble in one transparent surface.
5. Run `python -m pytest -q`.

For replacing an existing style, update `reference.png` and `character.png`, rebuild `box.png`, then restart the APP.

## Visual Check

Open the desktop APP, choose the style in Settings, trigger a suggestion, and check:

- the character appears close to the suggestion bubble,
- the text remains readable,
- `top-left` and `bottom-right` anchor by the visible bubble rather than by transparent character padding,
- the bubble stays inside the monitor bounds,
- `Tab` accept and `Esc` reject still work.
