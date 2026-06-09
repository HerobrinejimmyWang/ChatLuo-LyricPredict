from __future__ import annotations

import argparse
from pathlib import Path

from PySide6 import QtCore, QtGui


def image_stats(path: Path) -> dict[str, int | bool]:
    original = QtGui.QImage(str(path))
    has_alpha = original.hasAlphaChannel()
    image = original.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    transparent = 0
    for row in range(image.height()):
        for col in range(image.width()):
            if image.pixelColor(col, row).alpha() == 0:
                transparent += 1
    return {
        "width": image.width(),
        "height": image.height(),
        "has_alpha": has_alpha,
        "transparent": transparent,
    }


def make_character_layer(
    source_path: Path,
    output_path: Path,
    crop: tuple[int, int, int, int],
    white_threshold: int = 245,
) -> None:
    image = QtGui.QImage(str(source_path))
    if image.isNull():
        raise ValueError(f"Cannot read image: {source_path}")

    x, y, width, height = crop
    cropped = image.copy(x, y, min(width, image.width() - x), min(height, image.height() - y))
    cropped = cropped.convertToFormat(QtGui.QImage.Format.Format_ARGB32)

    for row in range(cropped.height()):
        for col in range(cropped.width()):
            color = QtGui.QColor(cropped.pixel(col, row))
            if color.red() > white_threshold and color.green() > white_threshold and color.blue() > white_threshold:
                color.setAlpha(0)
                cropped.setPixelColor(col, row, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cropped.save(str(output_path)):
        raise ValueError(f"Cannot save image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LyricPredict suggestion style assets.")
    parser.add_argument("--source", required=True, help="Reference PNG path.")
    parser.add_argument("--out-dir", required=True, help="Suggestion style asset directory.")
    parser.add_argument("--crop", nargs=4, type=int, default=(0, 0, 760, 430), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--white-threshold", type=int, default=245)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    reference = out_dir / "reference.png"
    character = out_dir / "character.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(source.read_bytes())
    make_character_layer(reference, character, tuple(args.crop), args.white_threshold)
    print(f"Wrote {reference}")
    print(f"Reference stats: {image_stats(reference)}")
    print(f"Wrote {character}")
    print(f"Character stats: {image_stats(character)}")


if __name__ == "__main__":
    main()
