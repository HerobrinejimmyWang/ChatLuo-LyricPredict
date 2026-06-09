from __future__ import annotations

from pathlib import Path
import sys

from PySide6 import QtCore, QtGui


def _rounded_path(rect: QtCore.QRectF, radius: float) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def build_luo_box(out_dir: Path) -> None:
    reference = out_dir / "reference.png"
    character_path = out_dir / "character.png"
    box_path = out_dir / "box.png"
    character = QtGui.QPixmap(str(character_path))
    if character.isNull():
        raise ValueError(f"Cannot read character layer: {character_path}")

    canvas_width = 960
    canvas_height = 340
    image = QtGui.QImage(canvas_width, canvas_height, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

    bubble = QtCore.QRectF(34, 178, canvas_width - 68, 162)
    outer_path = _rounded_path(bubble, 28)

    painter.setPen(QtGui.QPen(QtGui.QColor("#7767d8"), 4))
    painter.setBrush(QtGui.QColor("#fffefe"))
    painter.drawPath(outer_path)

    highlight_pen = QtGui.QPen(QtGui.QColor(190, 185, 255, 170), 8)
    painter.setPen(highlight_pen)
    painter.drawRoundedRect(bubble.adjusted(8, 8, -8, -8), 22, 22)

    dash_pen = QtGui.QPen(QtGui.QColor("#7767d8"), 2)
    dash_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
    painter.setPen(dash_pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(bubble.adjusted(30, 28, -30, -28), 18, 18)

    accent_pen = QtGui.QPen(QtGui.QColor("#2b0d3b"), 2)
    accent_brushes = [QtGui.QColor("#c8f5ff"), QtGui.QColor("#fff4a0"), QtGui.QColor("#f5b4e1")]
    painter.setPen(accent_pen)
    for rect, color, angle in [
        (QtCore.QRectF(18, 214, 40, 40), accent_brushes[0], -14),
        (QtCore.QRectF(54, 260, 28, 28), accent_brushes[1], 8),
        (QtCore.QRectF(canvas_width - 82, 222, 34, 34), accent_brushes[2], -18),
    ]:
        painter.save()
        painter.translate(rect.center())
        painter.rotate(angle)
        painter.setBrush(color)
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(-rect.width() / 2, rect.height() / 2),
                    QtCore.QPointF(0, -rect.height() / 2),
                    QtCore.QPointF(rect.width() / 2, rect.height() / 2),
                ]
            )
        )
        painter.restore()

    character_scaled = character.scaledToWidth(350, QtCore.Qt.TransformationMode.SmoothTransformation)
    painter.drawPixmap(72, 0, character_scaled)
    painter.end()

    out_dir.mkdir(parents=True, exist_ok=True)
    if not image.save(str(box_path)):
        raise ValueError(f"Cannot save box image: {box_path}")
    print(f"Wrote {box_path}")
    print(f"Source reference: {reference}")


def main() -> None:
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(sys.argv)
    build_luo_box(Path("assets/suggestion_styles/luo"))
    app.quit()


if __name__ == "__main__":
    main()
