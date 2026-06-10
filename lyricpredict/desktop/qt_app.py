from __future__ import annotations

import ctypes
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from ctypes import wintypes

from lyricpredict.config import load_config
from lyricpredict.generation import LyricGenerator, normalize_prediction_boundary
from lyricpredict.importer import prepare_dataset

from .auto_read import AutoReadCounter, auto_read_scope_allows, is_text_change_key
from .app_state import DesktopAppState, load_desktop_app_state, save_desktop_app_state
from .model_registry import (
    ModelProfile,
    ModelRegistry,
    copy_files_to_profile,
    create_model_profile,
    create_runtime_config,
    load_model_registry,
    save_model_registry,
)
from .settings import AppSettings, VALID_HOTKEYS, load_app_settings, save_app_settings
from .windows_io import (
    DesktopDependencyError,
    ULONG_PTR,
    cursor_position,
    focus_window,
    foreground_window,
    paste_text_with_restore,
    read_selected_text_with_restore,
    replace_previous_text,
    window_signature,
)
from .uia_context import read_uia_context_with_reason
from .workflow import (
    SuggestionKind,
    SuggestionPayload,
    SuggestionState,
    compose_insert_text,
    describe_correction,
    prediction_to_payload,
    trim_context,
)


def _load_qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise DesktopDependencyError("PySide6 is required for the Windows desktop app.") from exc
    return QtCore, QtGui, QtWidgets


def _parse_hotkey(hotkey: str) -> tuple[int, int]:
    modifiers = 0
    key = ""
    named_keys = {"space": 0x20, "tab": 0x09, "escape": 0x1B, "esc": 0x1B}
    for part in hotkey.split("+"):
        token = part.strip().lower()
        if token == "alt":
            modifiers |= 0x0001
        elif token in {"ctrl", "control"}:
            modifiers |= 0x0002
        elif token == "shift":
            modifiers |= 0x0004
        elif token == "win":
            modifiers |= 0x0008
        elif token:
            key = token
    if key in named_keys:
        return modifiers, named_keys[key]
    if len(key) != 1:
        raise ValueError(f"Unsupported hotkey key: {hotkey}")
    return modifiers, ord(key.upper())


def _build_tray_icon(QtGui) -> object:
    pixmap = QtGui.QPixmap(32, 32)
    pixmap.fill(QtGui.QColor("#1f7a68"))
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(QtGui.QColor("#fffdf9"), 3))
    painter.drawLine(9, 9, 9, 23)
    painter.drawLine(9, 23, 23, 23)
    painter.drawLine(15, 9, 23, 17)
    painter.end()
    return QtGui.QIcon(pixmap)


def run_desktop_app(settings_path: str | Path = "configs/app.yaml") -> int:
    QtCore, QtGui, QtWidgets = _load_qt()

    class HotkeyThread(QtCore.QThread):
        activated = QtCore.Signal(object)
        registered = QtCore.Signal(str)
        registration_failed = QtCore.Signal(str)

        def __init__(self, hotkey: str):
            super().__init__()
            self.hotkey = hotkey
            self.thread_id = 0

        def run(self) -> None:
            if not hasattr(ctypes, "windll"):
                message = "Windows hotkey API is unavailable."
                print(message, flush=True)
                self.registration_failed.emit(message)
                return
            user32 = ctypes.windll.user32
            self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
            try:
                modifiers, vk = _parse_hotkey(self.hotkey)
            except ValueError as exc:
                message = str(exc)
                print(message, flush=True)
                self.registration_failed.emit(message)
                return
            hotkey_id = 0x4C50
            if not user32.RegisterHotKey(None, hotkey_id, modifiers, vk):
                message = f"Hotkey unavailable: {self.hotkey}"
                print(message, flush=True)
                self.registration_failed.emit(message)
                return
            message = f"Hotkey ready: {self.hotkey}"
            print(message, flush=True)
            self.registered.emit(message)
            try:
                msg = wintypes.MSG()
                while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                    if msg.message == 0x0312 and msg.wParam == hotkey_id:
                        self.activated.emit(int(user32.GetForegroundWindow()))
            finally:
                user32.UnregisterHotKey(None, hotkey_id)

        def stop(self) -> None:
            if self.thread_id and hasattr(ctypes, "windll"):
                ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)

    class SuggestionKeyThread(QtCore.QThread):
        accepted = QtCore.Signal()
        rejected = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.thread_id = 0

        def run(self) -> None:
            if not hasattr(ctypes, "windll"):
                return
            user32 = ctypes.windll.user32
            self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
            tab_id = 0x4C51
            esc_id = 0x4C52
            registered_tab = bool(user32.RegisterHotKey(None, tab_id, 0, 0x09))
            registered_esc = bool(user32.RegisterHotKey(None, esc_id, 0, 0x1B))
            try:
                msg = wintypes.MSG()
                while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                    if msg.message != 0x0312:
                        continue
                    if msg.wParam == tab_id:
                        self.accepted.emit()
                    elif msg.wParam == esc_id:
                        self.rejected.emit()
            finally:
                if registered_tab:
                    user32.UnregisterHotKey(None, tab_id)
                if registered_esc:
                    user32.UnregisterHotKey(None, esc_id)

        def stop(self) -> None:
            if self.thread_id and hasattr(ctypes, "windll"):
                ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)

    class AutoReadHookThread(QtCore.QThread):
        text_changed = QtCore.Signal(object)
        registered = QtCore.Signal(str)
        registration_failed = QtCore.Signal(str)

        def __init__(self):
            super().__init__()
            self.thread_id = 0
            self.hook = None
            self.callback = None

        def run(self) -> None:
            try:
                if not hasattr(ctypes, "windll"):
                    message = "Auto read hook unavailable."
                    print(message, flush=True)
                    self.registration_failed.emit(message)
                    return
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32
                user32.SetWindowsHookExW.restype = wintypes.HANDLE
                user32.SetWindowsHookExW.argtypes = [
                    ctypes.c_int,
                    ctypes.c_void_p,
                    wintypes.HINSTANCE,
                    wintypes.DWORD,
                ]
                user32.CallNextHookEx.restype = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
                user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
                kernel32.GetModuleHandleW.restype = wintypes.HMODULE
                self.thread_id = kernel32.GetCurrentThreadId()
                low_level_keyboard_proc = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

                class KBDLLHOOKSTRUCT(ctypes.Structure):
                    _fields_ = [
                        ("vkCode", wintypes.DWORD),
                        ("scanCode", wintypes.DWORD),
                        ("flags", wintypes.DWORD),
                        ("time", wintypes.DWORD),
                        ("dwExtraInfo", ULONG_PTR),
                    ]

                def callback(n_code, w_param, l_param):
                    if n_code == 0 and int(w_param) in {0x0100, 0x0104}:
                        event = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                        ctrl_pressed = bool(user32.GetAsyncKeyState(0x11) & 0x8000)
                        alt_pressed = bool(user32.GetAsyncKeyState(0x12) & 0x8000)
                        if is_text_change_key(int(event.vkCode), ctrl_pressed=ctrl_pressed, alt_pressed=alt_pressed):
                            hwnd = int(user32.GetForegroundWindow())
                            print(f"Auto read key event: hwnd={hwnd} vk={int(event.vkCode)}", flush=True)
                            self.text_changed.emit(hwnd)
                    return user32.CallNextHookEx(self.hook, n_code, w_param, l_param)

                self.callback = low_level_keyboard_proc(callback)
                self.hook = user32.SetWindowsHookExW(13, self.callback, kernel32.GetModuleHandleW(None), 0)
                if not self.hook:
                    message = f"Auto read hook failed: {ctypes.get_last_error()}"
                    print(message, flush=True)
                    self.registration_failed.emit(message)
                    return
                message = "Auto read hook ready"
                print(message, flush=True)
                self.registered.emit(message)
                msg = wintypes.MSG()
                while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
            except Exception as exc:
                message = f"Auto read hook error: {exc}"
                print(message, flush=True)
                self.registration_failed.emit(message)
            finally:
                if self.hook:
                    ctypes.windll.user32.UnhookWindowsHookEx(self.hook)
                    self.hook = None

        def stop(self) -> None:
            if self.thread_id and hasattr(ctypes, "windll"):
                ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)

    class ModelBuildThread(QtCore.QThread):
        completed = QtCore.Signal(str)
        failed = QtCore.Signal(str)

        def __init__(self, config_path: str, build_kind: str = "ngram"):
            super().__init__()
            self.config_path = config_path
            self.build_kind = build_kind

        def run(self) -> None:
            if self.build_kind == "transformer":
                commands = [
                    [sys.executable, "-m", "lyricpredict.prepare", "--config", self.config_path],
                    [sys.executable, "-m", "lyricpredict.train", "--config", self.config_path],
                    [sys.executable, "-m", "lyricpredict.calibrate", "--config", self.config_path],
                ]
                completed_message = "Transformer training finished"
            else:
                commands = [
                    [sys.executable, "-m", "lyricpredict.prepare", "--config", self.config_path],
                    [sys.executable, "-m", "lyricpredict.ngram_model", "--config", self.config_path, "--order", "32"],
                ]
                completed_message = "Model ngram rebuilt"
            try:
                for command in commands:
                    result = subprocess.run(command, cwd=Path.cwd(), capture_output=True, text=True, check=False)
                    if result.returncode != 0:
                        message = (result.stderr or result.stdout or "model build failed").strip()
                        self.failed.emit(message[-500:])
                        return
                self.completed.emit(completed_message)
            except Exception as exc:
                self.failed.emit(str(exc))

    class LuoSuggestionCanvas(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.text = ""
            self.meta = ""
            self.canvas_width = 720
            self.box_pixmap: QtGui.QPixmap | None = None
            self.character_pixmap: QtGui.QPixmap | None = None
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAutoFillBackground(False)

        def _character(self) -> object | None:
            if self.character_pixmap is not None:
                return self.character_pixmap
            style_dir = Path.cwd() / "assets" / "suggestion_styles" / "luo"
            pixmap = QtGui.QPixmap(str(style_dir / "character.png"))
            if pixmap.isNull():
                reference = QtGui.QPixmap(str(style_dir / "reference.png"))
                if reference.isNull():
                    return None
                pixmap = reference.copy(0, 0, min(760, reference.width()), min(430, reference.height()))
            self.character_pixmap = pixmap
            return self.character_pixmap

        def _box(self) -> object | None:
            if self.box_pixmap is not None:
                return self.box_pixmap
            style_dir = Path.cwd() / "assets" / "suggestion_styles" / "luo"
            pixmap = QtGui.QPixmap(str(style_dir / "box.png"))
            if pixmap.isNull():
                return None
            self.box_pixmap = pixmap
            return self.box_pixmap

        def set_content(self, text: str, meta: str, width: int) -> None:
            self.text = text
            self.meta = meta
            self.canvas_width = max(620, width)
            self.updateGeometry()
            self.update()

        def sizeHint(self) -> object:
            metrics = QtGui.QFontMetrics(self.font())
            base_height = round(self.canvas_width * 340 / 960)
            text_rect = metrics.boundingRect(
                QtCore.QRect(0, 0, self.canvas_width - 86, 420),
                QtCore.Qt.TextFlag.TextWordWrap,
                self.text or " ",
            )
            return QtCore.QSize(self.canvas_width, max(base_height, 172 + text_rect.height()))

        def paintEvent(self, event) -> None:
            del event
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            width = self.width()
            height = self.height()

            box = self._box()
            if box is not None:
                painter.drawPixmap(QtCore.QRect(0, 0, width, height), box)

            painter.setPen(QtGui.QColor("#2a2540"))
            painter.setFont(self.font())
            text_y = round(height * 0.65)
            painter.drawText(
                QtCore.QRectF(60, text_y, width - 382, max(34, height - text_y - 62)),
                QtCore.Qt.TextFlag.TextWordWrap,
                self.text,
            )

            meta_font = QtGui.QFont(self.font())
            meta_font.setPointSize(max(8, meta_font.pointSize() - 1))
            painter.setFont(meta_font)
            painter.setPen(QtGui.QColor("#6f66a8"))
            painter.drawText(QtCore.QRectF(60, height - 46, width - 382, 24), self.meta)

    class SuggestionLine(QtWidgets.QWidget):
        accepted = QtCore.Signal()
        rejected = QtCore.Signal()
        ignored = QtCore.Signal()

        def __init__(self):
            super().__init__(
                None,
                QtCore.Qt.WindowType.Tool
                | QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint,
            )
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAutoFillBackground(False)
            self.setObjectName("suggestionLine")
            self.setWindowTitle("LyricPredict Suggestion")
            layout = QtWidgets.QStackedLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self.character_label = QtWidgets.QLabel("")
            self.character_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.character_label.hide()
            self.bubble = QtWidgets.QWidget()
            self.bubble.setObjectName("bubble")
            self.bubble.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.bubble.setAutoFillBackground(False)
            bubble_layout = QtWidgets.QVBoxLayout(self.bubble)
            bubble_layout.setContentsMargins(14, 10, 14, 10)
            self.text_label = QtWidgets.QLabel("")
            self.text_label.setWordWrap(True)
            self.meta_label = QtWidgets.QLabel("Tab accept | Esc reject")
            self.meta_label.setObjectName("meta")
            bubble_layout.addWidget(self.text_label)
            bubble_layout.addWidget(self.meta_label)
            layout.addWidget(self.bubble)
            self.luo_canvas = LuoSuggestionCanvas()
            layout.addWidget(self.luo_canvas)
            self.stack = layout
            self.timer = QtCore.QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self._timeout)
            self.character_pixmap: QtGui.QPixmap | None = None
            self.current_style = "plain"
            self._apply_style("plain")

        def _luo_character(self) -> object | None:
            if self.character_pixmap is not None:
                return self.character_pixmap
            style_dir = Path.cwd() / "assets" / "suggestion_styles" / "luo"
            character = style_dir / "character.png"
            pixmap = QtGui.QPixmap(str(character))
            if pixmap.isNull():
                reference = style_dir / "reference.png"
                pixmap = QtGui.QPixmap(str(reference))
                if pixmap.isNull():
                    return None
                pixmap = pixmap.copy(0, 0, min(760, pixmap.width()), min(430, pixmap.height()))
            self.character_pixmap = pixmap.scaledToWidth(150, QtCore.Qt.TransformationMode.SmoothTransformation)
            return self.character_pixmap
            source = Path.cwd() / "建议框风格1-LUO.png"
            source = Path.cwd() / "assets" / "suggestion_styles" / "luo" / "reference.png"
            if not source.exists():
                source = Path.cwd() / "建议框风格1-LUO.png"
            if not source.exists():
                return None
            pixmap = QtGui.QPixmap(str(source))
            if pixmap.isNull():
                return None
            cropped = pixmap.copy(0, 0, min(760, pixmap.width()), min(430, pixmap.height())).scaledToWidth(
                150, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            image = cropped.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
            for y in range(image.height()):
                for x in range(image.width()):
                    color = QtGui.QColor(image.pixel(x, y))
                    if color.red() > 245 and color.green() > 245 and color.blue() > 245:
                        color.setAlpha(0)
                        image.setPixelColor(x, y, color)
            self.character_pixmap = QtGui.QPixmap.fromImage(image)
            return self.character_pixmap

        def _apply_style(self, style: str) -> None:
            self.current_style = style
            if style == "luo":
                self.character_label.hide()
                self.stack.setCurrentWidget(self.luo_canvas)
                self.setStyleSheet(
                    "#suggestionLine { background-color: transparent; color: #2a2540; }"
                )
                return
            self.character_label.hide()
            self.stack.setCurrentWidget(self.bubble)
            self.setStyleSheet(
                "#suggestionLine { background-color: transparent; color: #1d2329; }"
                "#bubble { background: #fffdf9; border: 1px solid #1f7a68; border-radius: 8px; }"
                "#meta { color: #68717b; font-size: 12px; }"
            )

        def show_payload(
            self,
            payload: SuggestionPayload,
            timeout_seconds: float,
            position: str = "bottom-right",
            style: str = "plain",
        ) -> None:
            self._apply_style(style)
            corrected = f" | corrected" if payload.corrected_context else ""
            self.text_label.setText(payload.text)
            label = {
                SuggestionKind.PREDICTION: "Predict",
                SuggestionKind.CORRECTION: "Correct",
                SuggestionKind.CONTINUE: "Continue",
            }.get(payload.kind, "Suggest")
            self.meta_label.setText(
                f"{label} · {payload.reason} · {payload.confidence:.3f}{corrected} · Tab accept · Esc reject"
            )
            meta_text = self.meta_label.text()
            pos = cursor_position()
            if pos:
                margin = 8
                pointer_gap = 34
                point = QtCore.QPoint(pos[0], pos[1])
                screen = QtGui.QGuiApplication.screenAt(point) or QtGui.QGuiApplication.primaryScreen()
                if screen:
                    rect = screen.availableGeometry()
                    width = min(720 if style == "luo" else 480, max(260, rect.width() - margin * 2))
                else:
                    rect = None
                    width = 720 if style == "luo" else 480
                if style == "luo":
                    width = min(720, max(620, width))
                    self.luo_canvas.set_content(payload.text, meta_text, width)
                    self.setFixedSize(self.luo_canvas.sizeHint())
                else:
                    self.setFixedWidth(width)
                    self.setMinimumHeight(0)
                    self.setMaximumHeight(16777215)
                self.adjustSize()
                if style == "luo":
                    scale_x = self.width() / 960
                    scale_y = self.height() / 340
                    visual_left = round(34 * scale_x)
                    visual_top = round(148 * scale_y)
                    visual_right = round((34 + 892) * scale_x)
                    visual_bottom = round((148 + 158) * scale_y)
                else:
                    visual_left = 0
                    visual_top = 0
                    visual_right = self.width()
                    visual_bottom = self.height()
                if position == "top-left":
                    x = pos[0] - visual_right - pointer_gap
                    y = pos[1] - visual_bottom - pointer_gap
                else:
                    x = pos[0] - visual_left + pointer_gap
                    y = pos[1] - visual_top + pointer_gap
                if rect:
                    x = max(rect.left() + margin, min(x, rect.right() - self.width() - margin))
                    y = max(rect.top() + margin, min(y, rect.bottom() - self.height() - margin))
                self.move(x, y)
            else:
                if style == "luo":
                    self.luo_canvas.set_content(payload.text, meta_text, 620)
                    self.setFixedSize(self.luo_canvas.sizeHint())
                else:
                    self.resize(480, self.sizeHint().height())
            self.show()
            self.raise_()
            self.timer.start(int(timeout_seconds * 1000))

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key.Key_Tab:
                self.timer.stop()
                self.hide()
                self.accepted.emit()
                return
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.timer.stop()
                self.hide()
                self.rejected.emit()
                return
            self.timer.stop()
            self.hide()
            self.ignored.emit()

        def _timeout(self) -> None:
            if self.isVisible():
                self.hide()
                self.ignored.emit()

    class SettingsWindow(QtWidgets.QWidget):
        trigger_requested = QtCore.Signal()
        settings_changed = QtCore.Signal(object)
        import_requested = QtCore.Signal(list)
        model_selected = QtCore.Signal(str)
        new_model_requested = QtCore.Signal()
        rebuild_model_requested = QtCore.Signal()
        train_model_requested = QtCore.Signal()

        def __init__(self, settings: AppSettings, registry: ModelRegistry):
            super().__init__()
            self.setWindowTitle("LyricPredict Desktop")
            self.settings = settings
            self.registry = registry
            layout = QtWidgets.QVBoxLayout(self)

            self.enabled = QtWidgets.QCheckBox("Enable")
            self.enabled.setChecked(settings.enabled)
            layout.addWidget(self.enabled)

            self.model_combo = QtWidgets.QComboBox()
            layout.addWidget(QtWidgets.QLabel("Model"))
            layout.addWidget(self.model_combo)
            model_buttons = QtWidgets.QHBoxLayout()
            self.new_model_button = QtWidgets.QPushButton("New model")
            self.rebuild_model_button = QtWidgets.QPushButton("Rebuild ngram")
            self.train_model_button = QtWidgets.QPushButton("Train transformer")
            model_buttons.addWidget(self.new_model_button)
            model_buttons.addWidget(self.rebuild_model_button)
            model_buttons.addWidget(self.train_model_button)
            layout.addLayout(model_buttons)
            self.set_models(registry, settings.active_model_id)

            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(["auto", "model-only"])
            self.mode.setCurrentText(settings.mode)
            layout.addWidget(QtWidgets.QLabel("Mode"))
            layout.addWidget(self.mode)

            self.strictness = QtWidgets.QComboBox()
            self.strictness.addItems(["strict", "balanced", "tolerant"])
            self.strictness.setCurrentText(settings.strictness)
            layout.addWidget(QtWidgets.QLabel("Strictness"))
            layout.addWidget(self.strictness)

            self.context_window = QtWidgets.QComboBox()
            self.context_window.addItems(["10", "16", "24", "32"])
            self.context_window.setCurrentText(str(settings.context_window))
            layout.addWidget(QtWidgets.QLabel("Context window"))
            layout.addWidget(self.context_window)

            self.suggestion_position = QtWidgets.QComboBox()
            self.suggestion_position.addItems(["bottom-right", "top-left"])
            self.suggestion_position.setCurrentText(settings.suggestion_position)
            layout.addWidget(QtWidgets.QLabel("Suggestion position"))
            layout.addWidget(self.suggestion_position)

            self.suggestion_style = QtWidgets.QComboBox()
            self.suggestion_style.addItems(["plain", "luo"])
            self.suggestion_style.setCurrentText(settings.suggestion_style)
            layout.addWidget(QtWidgets.QLabel("Suggestion style"))
            layout.addWidget(self.suggestion_style)

            self.hotkey = QtWidgets.QComboBox()
            self.hotkey.addItems(list(VALID_HOTKEYS))
            self.hotkey.setCurrentText(settings.hotkey)
            layout.addWidget(QtWidgets.QLabel("Trigger hotkey"))
            layout.addWidget(self.hotkey)

            self.read_frequency = QtWidgets.QComboBox()
            self.read_frequency.addItems(["often", "sometimes", "seldom"])
            self.read_frequency.setCurrentText(settings.read_frequency)
            layout.addWidget(QtWidgets.QLabel("Read frequency"))
            layout.addWidget(self.read_frequency)

            self.auto_read_enabled = QtWidgets.QCheckBox("Enable auto read")
            self.auto_read_enabled.setChecked(settings.auto_read_enabled)
            self.auto_read_scope = QtWidgets.QComboBox()
            self.auto_read_scope.addItems(["used-windows", "all-windows"])
            self.auto_read_scope.setCurrentText(settings.auto_read_scope)
            layout.addWidget(self.auto_read_enabled)
            layout.addWidget(QtWidgets.QLabel("Auto read scope"))
            layout.addWidget(self.auto_read_scope)

            self.correction = QtWidgets.QCheckBox("Enable correction")
            self.correction.setChecked(settings.correction)
            self.include_separator = QtWidgets.QCheckBox("Include separator")
            self.include_separator.setChecked(settings.include_separator)
            layout.addWidget(self.correction)
            layout.addWidget(self.include_separator)

            self.default_separator = QtWidgets.QComboBox()
            for label, value in [("，", "，"), ("。", "。"), ("Space", " "), ("\\n", "\n")]:
                self.default_separator.addItem(label, value)
            separator_index = self.default_separator.findData(settings.default_separator)
            self.default_separator.setCurrentIndex(separator_index if separator_index >= 0 else 0)
            layout.addWidget(QtWidgets.QLabel("Default separator"))
            layout.addWidget(self.default_separator)

            self.test_context = QtWidgets.QPlainTextEdit()
            self.test_context.setPlaceholderText("Optional fallback context when no external selection is available")
            layout.addWidget(self.test_context)

            buttons = QtWidgets.QHBoxLayout()
            self.trigger_button = QtWidgets.QPushButton("Predict")
            self.import_button = QtWidgets.QPushButton("Import lyrics")
            buttons.addWidget(self.trigger_button)
            buttons.addWidget(self.import_button)
            layout.addLayout(buttons)

            self.status = QtWidgets.QLabel("Ready")
            layout.addWidget(self.status)

            for widget in [self.enabled, self.auto_read_enabled, self.correction, self.include_separator]:
                widget.stateChanged.connect(self._emit_settings)
            for widget in [
                self.mode,
                self.strictness,
                self.context_window,
                self.suggestion_position,
                self.suggestion_style,
                self.hotkey,
                self.read_frequency,
                self.auto_read_scope,
                self.default_separator,
            ]:
                widget.currentTextChanged.connect(self._emit_settings)
            self.model_combo.currentIndexChanged.connect(self._emit_model_selection)
            self.new_model_button.clicked.connect(self.new_model_requested.emit)
            self.rebuild_model_button.clicked.connect(self.rebuild_model_requested.emit)
            self.train_model_button.clicked.connect(self.train_model_requested.emit)
            self.trigger_button.clicked.connect(self.trigger_requested.emit)
            self.import_button.clicked.connect(self._choose_import_files)

        def set_models(self, registry: ModelRegistry, active_model_id: str) -> None:
            self.registry = registry
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            for profile in registry.models:
                self.model_combo.addItem(profile.name, profile.id)
            index = self.model_combo.findData(active_model_id)
            self.model_combo.setCurrentIndex(index if index >= 0 else 0)
            self.model_combo.blockSignals(False)

        def current_settings(self) -> AppSettings:
            separator = self.default_separator.currentData()
            return replace(
                self.settings,
                enabled=self.enabled.isChecked(),
                mode=self.mode.currentText(),
                strictness=self.strictness.currentText(),
                context_window=int(self.context_window.currentText()),
                suggestion_position=self.suggestion_position.currentText(),
                suggestion_style=self.suggestion_style.currentText(),
                hotkey=self.hotkey.currentText(),
                read_frequency=self.read_frequency.currentText(),
                auto_read_enabled=self.auto_read_enabled.isChecked(),
                auto_read_scope=self.auto_read_scope.currentText(),
                correction=self.correction.isChecked(),
                include_separator=self.include_separator.isChecked(),
                default_separator=separator,
                active_model_id=self.model_combo.currentData() or self.settings.active_model_id,
            )

        def _emit_settings(self, *_args) -> None:
            self.settings = self.current_settings()
            self.settings_changed.emit(self.settings)

        def _emit_model_selection(self, *_args) -> None:
            model_id = self.model_combo.currentData()
            if model_id:
                self.model_selected.emit(str(model_id))

        def _choose_import_files(self) -> None:
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Import lyrics",
                "",
                "Lyrics (*.txt *.lrc)",
            )
            if files:
                self.import_requested.emit(files)

    class Controller(QtCore.QObject):
        def __init__(self, app, settings: AppSettings, settings_path: Path):
            super().__init__()
            self.app = app
            self.settings = settings
            self.settings_path = settings_path
            self.state_path = settings_path.with_name("app_state.yaml")
            self.app_state = load_desktop_app_state(self.state_path)
            self.registry = load_model_registry(settings.model_registry_path)
            active_profile = self.registry.profile(settings.active_model_id)
            if active_profile is not None:
                self.settings = replace(
                    self.settings,
                    active_model_id=active_profile.id,
                    lyric_config_path=active_profile.config_path,
                )
            self.state = SuggestionState()
            self.target_hwnd: int | None = None
            self.active_context = ""
            self.generator: LyricGenerator | None = None
            self.prefetched_payload: SuggestionPayload | None = None
            self.prefetch_base_context = ""
            self.suggestion_key_thread: SuggestionKeyThread | None = None
            self.auto_read_thread = AutoReadHookThread()
            self.auto_read_thread.text_changed.connect(self.on_auto_read_key_event)
            self.auto_read_counter = AutoReadCounter(self.settings.read_change_threshold)
            self.auto_read_timer = QtCore.QTimer(self)
            self.auto_read_timer.setSingleShot(True)
            self.auto_read_timer.timeout.connect(self.try_auto_read)
            self.auto_read_hwnd: int | None = None
            self.used_windows: set[int] = set()
            self.used_window_signatures: set[str] = set(self.app_state.auto_read_used_window_signatures)
            self.last_auto_context = ""
            self.model_build_thread: ModelBuildThread | None = None
            self.window = SettingsWindow(self.settings, self.registry)
            self.suggestion = SuggestionLine()
            self.auto_read_thread.registered.connect(self.set_auto_read_status)
            self.auto_read_thread.registration_failed.connect(self.set_auto_read_status)
            self.hotkey_thread = HotkeyThread(self.settings.hotkey)
            self.hotkey_thread.activated.connect(self.trigger_prediction_from_hotkey)
            self.hotkey_thread.registered.connect(self.window.status.setText)
            self.hotkey_thread.registration_failed.connect(self.window.status.setText)
            self.window.trigger_requested.connect(self.trigger_prediction)
            self.window.settings_changed.connect(self.update_settings)
            self.window.import_requested.connect(self.import_files)
            self.window.model_selected.connect(self.select_model)
            self.window.new_model_requested.connect(self.create_model)
            self.window.rebuild_model_requested.connect(self.rebuild_model)
            self.window.train_model_requested.connect(self.train_model)
            self.suggestion.accepted.connect(self.accept_suggestion)
            self.suggestion.rejected.connect(self.reject_suggestion)
            self.suggestion.ignored.connect(self.ignore_suggestion)
            self.tray = QtWidgets.QSystemTrayIcon(_build_tray_icon(QtGui), self.app)
            menu = QtWidgets.QMenu()
            show_action = menu.addAction("Settings")
            predict_action = menu.addAction("Predict now")
            quit_action = menu.addAction("Quit")
            show_action.triggered.connect(self.window.show)
            predict_action.triggered.connect(self.trigger_prediction)
            quit_action.triggered.connect(self.quit)
            self.tray.setContextMenu(menu)
            self.tray.setToolTip("LyricPredict")
            self.tray.show()

        def start(self) -> None:
            self.window.show()
            self.hotkey_thread.start()
            self.auto_read_thread.start()

        def set_auto_read_status(self, message: str) -> None:
            state = "enabled" if self.settings.auto_read_enabled else "disabled"
            suffix = ""
            if (
                self.settings.auto_read_enabled
                and self.settings.auto_read_scope == "used-windows"
                and not self.used_windows
                and not self.used_window_signatures
            ):
                suffix = " · press trigger hotkey once in target window"
            self.window.status.setText(f"{message} ({state}){suffix}")

        def remember_used_window(self, hwnd: int | None) -> None:
            if not hwnd:
                return
            self.used_windows.add(int(hwnd))
            try:
                signature = window_signature(hwnd)
            except DesktopDependencyError:
                signature = ""
            if not signature:
                return
            if signature not in self.used_window_signatures:
                self.used_window_signatures.add(signature)
                self.app_state = DesktopAppState(auto_read_used_window_signatures=self.used_window_signatures)
                save_desktop_app_state(self.app_state, self.state_path)
                print(f"Auto read remembered window: {signature}", flush=True)

        def auto_read_scope_permits(self, hwnd: int | None) -> bool:
            if auto_read_scope_allows(self.settings.auto_read_scope, hwnd, self.used_windows):
                return True
            if self.settings.auto_read_scope != "used-windows" or not hwnd:
                return False
            try:
                signature = window_signature(hwnd)
            except DesktopDependencyError:
                return False
            return bool(signature and signature in self.used_window_signatures)

        def trigger_prediction_from_hotkey(self, target_hwnd: int | None = None) -> None:
            self.target_hwnd = target_hwnd
            self.remember_used_window(target_hwnd)
            QtCore.QTimer.singleShot(180, self.trigger_prediction)

        def update_settings(self, settings: AppSettings) -> None:
            old_hotkey = self.settings.hotkey
            old_auto_read_enabled = self.settings.auto_read_enabled
            old_auto_read_scope = self.settings.auto_read_scope
            self.settings = settings
            self.auto_read_counter.threshold = settings.read_change_threshold
            save_app_settings(settings, self.settings_path)
            self.generator = None
            if (
                settings.auto_read_enabled != old_auto_read_enabled
                or settings.auto_read_scope != old_auto_read_scope
            ):
                state = "enabled" if settings.auto_read_enabled else "disabled"
                suffix = ""
                if settings.auto_read_enabled and settings.auto_read_scope == "used-windows" and not self.used_windows:
                    suffix = " · press trigger hotkey once in target window"
                self.window.status.setText(f"Auto read {state}: {settings.auto_read_scope}{suffix}")
            if settings.hotkey != old_hotkey:
                self.hotkey_thread.stop()
                self.hotkey_thread.wait(500)
                self.hotkey_thread = HotkeyThread(settings.hotkey)
                self.hotkey_thread.activated.connect(self.trigger_prediction_from_hotkey)
                self.hotkey_thread.registered.connect(self.window.status.setText)
                self.hotkey_thread.registration_failed.connect(self.window.status.setText)
                self.hotkey_thread.start()

        def _save_registry(self) -> None:
            save_model_registry(self.registry, self.settings.model_registry_path)

        def _active_profile(self) -> ModelProfile | None:
            return self.registry.profile(self.settings.active_model_id)

        def select_model(self, model_id: str) -> None:
            profile = self.registry.profile(model_id)
            if profile is None:
                self.window.status.setText(f"Unknown model: {model_id}")
                return
            self.registry = ModelRegistry(active_model=profile.id, models=self.registry.models)
            self._save_registry()
            self.settings = replace(self.settings, active_model_id=profile.id, lyric_config_path=profile.config_path)
            save_app_settings(self.settings, self.settings_path)
            self.window.settings = self.settings
            self.generator = None
            self.prefetched_payload = None
            self.window.status.setText(f"Model selected: {profile.name}")

        def create_model(self) -> None:
            name, ok = QtWidgets.QInputDialog.getText(self.window, "New model", "Model name:")
            if not ok or not name.strip():
                return
            profile = create_model_profile(name.strip(), self.registry)
            self.registry = ModelRegistry(active_model=profile.id, models=[*self.registry.models, profile])
            self._save_registry()
            self.settings = replace(self.settings, active_model_id=profile.id, lyric_config_path=profile.config_path)
            save_app_settings(self.settings, self.settings_path)
            self.window.set_models(self.registry, profile.id)
            self.window.settings = self.settings
            self.generator = None
            self.window.status.setText(f"Created model: {profile.name}")

        def _start_model_build(self, build_kind: str) -> None:
            profile = self._active_profile()
            if profile is None:
                self.window.status.setText("No active model")
                return
            if self.model_build_thread is not None and self.model_build_thread.isRunning():
                self.window.status.setText("Model build already running")
                return
            create_runtime_config(profile)
            if build_kind == "transformer":
                self.window.status.setText(f"Training transformer: {profile.name}")
            else:
                self.window.status.setText(f"Rebuilding ngram: {profile.name}")
            self.model_build_thread = ModelBuildThread(profile.config_path, build_kind)
            self.model_build_thread.completed.connect(self._model_build_completed)
            self.model_build_thread.failed.connect(self._model_build_failed)
            self.model_build_thread.start()

        def rebuild_model(self) -> None:
            self._start_model_build("ngram")

        def train_model(self) -> None:
            self._start_model_build("transformer")

        def _model_build_completed(self, message: str) -> None:
            self.generator = None
            self.window.status.setText(message)

        def _model_build_failed(self, message: str) -> None:
            self.window.status.setText(f"Build failed: {message}")

        def _generator(self) -> LyricGenerator:
            if self.generator is None:
                self.generator = LyricGenerator(load_config(self.settings.lyric_config_path), mode=self.settings.mode)
            return self.generator

        def _read_context(self) -> str:
            try:
                selected = read_selected_text_with_restore().strip()
            except DesktopDependencyError as exc:
                self.window.status.setText(str(exc))
                selected = ""
            if not selected:
                selected = self.window.test_context.toPlainText().strip()
            return trim_context(selected, self.settings.context_window)

        def _read_uia_context(self) -> str:
            selected, reason = read_uia_context_with_reason(self.settings.context_window)
            print(f"Auto read UIA: reason={reason} chars={len(selected)}", flush=True)
            return trim_context(selected, self.settings.context_window)

        def on_auto_read_key_event(self, hwnd: int | None) -> None:
            if not self.settings.enabled or not self.settings.auto_read_enabled:
                print("Auto read ignored: disabled", flush=True)
                self.auto_read_counter.reset()
                return
            if self.state.visible or self.suggestion.isVisible():
                print("Auto read ignored: suggestion visible", flush=True)
                return
            if not self.auto_read_scope_permits(hwnd):
                print(
                    f"Auto read ignored: scope={self.settings.auto_read_scope} hwnd={hwnd} used={sorted(self.used_windows)} signatures={len(self.used_window_signatures)}",
                    flush=True,
                )
                return
            if self.auto_read_counter.record_change():
                self.auto_read_hwnd = int(hwnd) if hwnd else None
                self.window.status.setText("Auto read pending")
                self.auto_read_timer.start(400)

        def try_auto_read(self) -> None:
            if not self.settings.enabled or not self.settings.auto_read_enabled:
                return
            if self.state.visible or self.suggestion.isVisible():
                return
            hwnd = foreground_window()
            if not self.auto_read_scope_permits(hwnd):
                return
            if self.auto_read_hwnd and hwnd and int(hwnd) != int(self.auto_read_hwnd):
                return
            context = self._read_uia_context()
            if not context or context == self.last_auto_context:
                print("Auto read skipped: no new UIA context", flush=True)
                return
            if self.state.is_rejected(context):
                print("Auto read skipped: rejected cooldown", flush=True)
                return
            self.target_hwnd = hwnd
            self.active_context = context
            self.prefetched_payload = None
            self.prefetch_base_context = ""
            prediction = self._generator().predict(
                context,
                strictness=self.settings.strictness,
                correction=self.settings.correction,
            )
            payload = prediction_to_payload(context, prediction)
            self.last_auto_context = context
            if payload is None:
                self.window.status.setText(f"Auto no output: {prediction.reason}")
                print(f"Auto prediction rejected: {prediction.reason}", flush=True)
                return
            self._show_payload(payload)

        def _show_payload(self, payload: SuggestionPayload) -> None:
            self._stop_suggestion_keys()
            self.state.show(payload)
            self.window.status.setText(f"Suggested: {payload.reason} {payload.confidence:.3f}")
            print(f"Prediction suggested: {payload.reason} {payload.confidence:.3f}", flush=True)
            display_payload = payload
            if payload.kind in {SuggestionKind.PREDICTION, SuggestionKind.CONTINUE}:
                display_payload = replace(payload, text=compose_insert_text(payload, self.settings, replace_context=False))
            self.suggestion.show_payload(
                display_payload,
                self.settings.suggestion_timeout_seconds,
                self.settings.suggestion_position,
                self.settings.suggestion_style,
            )
            self._start_suggestion_keys()
            if payload.kind in {SuggestionKind.PREDICTION, SuggestionKind.CONTINUE}:
                QtCore.QTimer.singleShot(0, lambda current=payload: self._prefetch_after_payload(current))

        def _start_suggestion_keys(self) -> None:
            self._stop_suggestion_keys()
            self.suggestion_key_thread = SuggestionKeyThread()
            self.suggestion_key_thread.accepted.connect(self.accept_suggestion)
            self.suggestion_key_thread.rejected.connect(self.reject_suggestion)
            self.suggestion_key_thread.start()

        def _stop_suggestion_keys(self) -> None:
            if self.suggestion_key_thread is not None:
                self.suggestion_key_thread.stop()
                self.suggestion_key_thread.wait(500)
                self.suggestion_key_thread = None

        def _correction_payload(self, accepted: SuggestionPayload, inserted_text: str) -> SuggestionPayload | None:
            if not self.settings.correction or not accepted.corrected_context or accepted.corrected_context == accepted.context:
                return None
            description = describe_correction(accepted.context, accepted.corrected_context)
            if not description:
                return None
            return SuggestionPayload(
                context=accepted.context,
                corrected_context=accepted.corrected_context,
                inserted_text=inserted_text,
                text=f"{description}\n{accepted.corrected_context}",
                confidence=accepted.confidence,
                reason="correction",
                kind=SuggestionKind.CORRECTION,
            )

        def _prefetch_after_payload(self, payload: SuggestionPayload) -> None:
            if not self.state.visible or self.state.payload != payload:
                return
            if payload.corrected_context and self.settings.correction:
                return
            inserted = compose_insert_text(payload, self.settings, replace_context=False)
            base_context = f"{payload.context}{inserted}"
            context = trim_context(base_context, self.settings.context_window)
            prediction = self._generator().predict(
                context,
                strictness=self.settings.strictness,
                correction=self.settings.correction,
            )
            next_payload = prediction_to_payload(context, prediction)
            if next_payload is None:
                self.prefetched_payload = None
                self.prefetch_base_context = ""
                return
            self.prefetched_payload = replace(next_payload, kind=SuggestionKind.CONTINUE)
            self.prefetch_base_context = base_context

        def _suggest_continue(self) -> None:
            if not self.active_context:
                return
            if self.prefetched_payload is not None and self.prefetch_base_context == self.active_context:
                payload = self.prefetched_payload
                self.prefetched_payload = None
                self.prefetch_base_context = ""
                self._show_payload(payload)
                return
            context = trim_context(self.active_context, self.settings.context_window)
            prediction = self._generator().predict(
                context,
                strictness=self.settings.strictness,
                correction=self.settings.correction,
            )
            payload = prediction_to_payload(context, prediction)
            if payload is None:
                self.window.status.setText(f"No continue: {prediction.reason}")
                print(f"Continue rejected: {prediction.reason}", flush=True)
                return
            payload = replace(payload, kind=SuggestionKind.CONTINUE)
            self._show_payload(payload)

        def trigger_prediction(self) -> None:
            print("Prediction requested", flush=True)
            if not self.settings.enabled:
                self.window.status.setText("Disabled")
                print("Prediction skipped: disabled", flush=True)
                return
            if self.target_hwnd:
                focus_window(self.target_hwnd)
            context = self._read_context()
            if not context:
                self.window.status.setText("No context selected")
                print("Prediction skipped: no context", flush=True)
                return
            self.target_hwnd = self.target_hwnd or foreground_window()
            self.remember_used_window(self.target_hwnd)
            if self.state.is_rejected(context):
                self.window.status.setText("Suppressed after reject")
                print("Prediction skipped: rejected cooldown", flush=True)
                return
            self.active_context = context
            self.prefetched_payload = None
            self.prefetch_base_context = ""
            prediction = self._generator().predict(
                context,
                strictness=self.settings.strictness,
                correction=self.settings.correction,
            )
            payload = prediction_to_payload(context, prediction)
            if payload is None:
                self.window.status.setText(f"No output: {prediction.reason}")
                print(f"Prediction rejected: {prediction.reason}", flush=True)
                return
            self._show_payload(payload)

        def accept_suggestion(self) -> None:
            self._stop_suggestion_keys()
            self.suggestion.timer.stop()
            self.suggestion.hide()
            payload = self.state.accept()
            if not payload:
                return
            if payload.kind == SuggestionKind.CORRECTION:
                replacement = f"{payload.corrected_context or payload.context}{normalize_prediction_boundary(payload.corrected_context or payload.context, payload.inserted_text)}"
                replace_length = len(payload.context + payload.inserted_text)
                try:
                    focus_window(self.target_hwnd)
                    replace_previous_text(replace_length, replacement, prefer_paste=True)
                    self.active_context = replacement
                    self.window.status.setText("Corrected")
                    print("Correction accepted", flush=True)
                except DesktopDependencyError as exc:
                    self.window.status.setText(f"Correction failed: {exc}")
                    print(f"Correction input failed: {exc}", flush=True)
                QtCore.QTimer.singleShot(120, self._suggest_continue)
                return

            text = compose_insert_text(payload, self.settings, replace_context=False)
            try:
                focus_window(self.target_hwnd)
                paste_text_with_restore(text)
                self.active_context = f"{payload.context}{text}"
                self.window.status.setText("Accepted")
                print("Suggestion accepted", flush=True)
                correction = self._correction_payload(payload, text)
                if correction:
                    QtCore.QTimer.singleShot(120, lambda: self._show_payload(correction))
                else:
                    QtCore.QTimer.singleShot(120, self._suggest_continue)
            except DesktopDependencyError as exc:
                self.window.status.setText(f"Input failed: {exc}")
                self.suggestion.text_label.setText(f"Copy fallback: {text}")
                self.suggestion.show()
                print(f"Suggestion input failed: {exc}", flush=True)

        def reject_suggestion(self) -> None:
            self._stop_suggestion_keys()
            self.suggestion.timer.stop()
            self.suggestion.hide()
            payload = self.state.payload
            if payload and payload.kind == SuggestionKind.CORRECTION:
                self.state.reject(cooldown_seconds=0)
                self.window.status.setText("Correction rejected")
                print("Correction rejected", flush=True)
                QtCore.QTimer.singleShot(120, self._suggest_continue)
                return
            self.state.reject()
            self.active_context = ""
            self.window.status.setText("Rejected")
            print("Suggestion rejected", flush=True)

        def ignore_suggestion(self) -> None:
            self._stop_suggestion_keys()
            self.suggestion.timer.stop()
            self.suggestion.hide()
            payload = self.state.payload
            self.state.ignore()
            if payload and payload.kind == SuggestionKind.CORRECTION:
                self.window.status.setText("Correction ignored")
                print("Correction ignored", flush=True)
                QtCore.QTimer.singleShot(120, self._suggest_continue)
                return
            self.active_context = ""
            self.window.status.setText("Ignored")
            print("Suggestion ignored", flush=True)

        def import_files(self, files: list[str]) -> None:
            profile = self._active_profile()
            if profile is None:
                self.window.status.setText("No active model")
                return
            copied = copy_files_to_profile(files, profile)
            config = load_config(profile.config_path)
            stats = prepare_dataset(config.paths.raw_dir, config.paths.processed_dir, config.training.validation_ratio)
            self.generator = None
            self.window.status.setText(f"Imported {copied} files, prepared {stats.lines} lines")

        def quit(self) -> None:
            self._stop_suggestion_keys()
            if self.model_build_thread is not None and self.model_build_thread.isRunning():
                self.model_build_thread.terminate()
                self.model_build_thread.wait(500)
            self.auto_read_thread.stop()
            self.auto_read_thread.wait(500)
            self.hotkey_thread.stop()
            self.hotkey_thread.wait(500)
            self.app.quit()

    settings_path = Path(settings_path)
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    controller = Controller(app, load_app_settings(settings_path), settings_path)
    controller.start()
    return app.exec()
