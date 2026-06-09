from __future__ import annotations

import ctypes
import time
import uuid
from ctypes import wintypes


class DesktopDependencyError(RuntimeError):
    pass


def _require_windows() -> None:
    if not hasattr(ctypes, "windll"):
        raise DesktopDependencyError("Windows desktop integration is only available on Windows.")


def _clipboard_text() -> str:
    try:
        import win32clipboard
        import win32con
    except ImportError as exc:
        raise DesktopDependencyError("pywin32 is required for clipboard text capture.") from exc

    win32clipboard.OpenClipboard()
    try:
        if not win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            return ""
        return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT) or ""
    finally:
        win32clipboard.CloseClipboard()


def _set_clipboard_text(text: str) -> None:
    try:
        import win32clipboard
        import win32con
    except ImportError as exc:
        raise DesktopDependencyError("pywin32 is required for clipboard text capture.") from exc

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def send_hotkey_ctrl_c(delay_seconds: float = 0.14) -> None:
    _require_windows()
    user32 = ctypes.windll.user32
    keybd_event = user32.keybd_event
    vk_control = 0x11
    vk_c = 0x43
    keyeventf_keyup = 0x0002
    keybd_event(vk_control, 0, 0, 0)
    keybd_event(vk_c, 0, 0, 0)
    keybd_event(vk_c, 0, keyeventf_keyup, 0)
    keybd_event(vk_control, 0, keyeventf_keyup, 0)
    time.sleep(delay_seconds)


def collapse_selection_to_end(delay_seconds: float = 0.03) -> None:
    _require_windows()
    user32 = ctypes.windll.user32
    keybd_event = user32.keybd_event
    vk_right = 0x27
    keyeventf_keyup = 0x0002
    keybd_event(vk_right, 0, 0, 0)
    keybd_event(vk_right, 0, keyeventf_keyup, 0)
    time.sleep(delay_seconds)


def read_selected_text_with_restore() -> str:
    original = _clipboard_text()
    sentinel = f"__LYRICPREDICT_EMPTY_SELECTION_{uuid.uuid4()}__"
    try:
        _set_clipboard_text(sentinel)
        send_hotkey_ctrl_c()
        copied = _clipboard_text()
        if copied == sentinel:
            return ""
        if copied:
            collapse_selection_to_end()
        return copied
    finally:
        _set_clipboard_text(original)


def send_hotkey_ctrl_v(delay_seconds: float = 0.08) -> None:
    _require_windows()
    user32 = ctypes.windll.user32
    keybd_event = user32.keybd_event
    vk_control = 0x11
    vk_v = 0x56
    keyeventf_keyup = 0x0002
    keybd_event(vk_control, 0, 0, 0)
    keybd_event(vk_v, 0, 0, 0)
    keybd_event(vk_v, 0, keyeventf_keyup, 0)
    keybd_event(vk_control, 0, keyeventf_keyup, 0)
    time.sleep(delay_seconds)


def paste_text_with_restore(text: str) -> None:
    if not text:
        return
    original = _clipboard_text()
    try:
        _set_clipboard_text(text)
        send_hotkey_ctrl_v()
    finally:
        _set_clipboard_text(original)


ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]


def _unicode_input(char: str, keyup: bool = False) -> INPUT:
    keyeventf_unicode = 0x0004
    keyeventf_keyup = 0x0002
    flags = keyeventf_unicode | (keyeventf_keyup if keyup else 0)
    return INPUT(type=1, union=INPUT_UNION(ki=KEYBDINPUT(0, ord(char), flags, 0, 0)))


def _virtual_key_input(vk: int, keyup: bool = False) -> INPUT:
    keyeventf_keyup = 0x0002
    return INPUT(type=1, union=INPUT_UNION(ki=KEYBDINPUT(vk, 0, keyeventf_keyup if keyup else 0, 0, 0)))


def _send_inputs(inputs: list[INPUT]) -> None:
    if not inputs:
        return
    user32 = ctypes.windll.user32
    array_type = INPUT * len(inputs)
    sent = user32.SendInput(len(inputs), array_type(*inputs), ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise DesktopDependencyError("SendInput did not accept the full input sequence.")


VK_RETURN = 0x0D


def _text_inputs(text: str) -> list[INPUT]:
    inputs: list[INPUT] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            inputs.append(_virtual_key_input(VK_RETURN, keyup=False))
            inputs.append(_virtual_key_input(VK_RETURN, keyup=True))
        elif char == "\n":
            inputs.append(_virtual_key_input(VK_RETURN, keyup=False))
            inputs.append(_virtual_key_input(VK_RETURN, keyup=True))
        else:
            inputs.append(_unicode_input(char, keyup=False))
            inputs.append(_unicode_input(char, keyup=True))
        index += 1
    return inputs


def type_unicode_text(text: str) -> None:
    _require_windows()
    if not text:
        return
    inputs = _text_inputs(text)
    _send_inputs(inputs)


def select_previous_chars(count: int, delay_seconds: float = 0.06) -> None:
    _require_windows()
    if count <= 0:
        return
    user32 = ctypes.windll.user32
    keybd_event = user32.keybd_event
    vk_shift = 0x10
    vk_left = 0x25
    keyeventf_keyup = 0x0002
    keybd_event(vk_shift, 0, 0, 0)
    try:
        for _ in range(count):
            keybd_event(vk_left, 0, 0, 0)
            time.sleep(0.004)
            keybd_event(vk_left, 0, keyeventf_keyup, 0)
            time.sleep(0.004)
    finally:
        keybd_event(vk_shift, 0, keyeventf_keyup, 0)
    time.sleep(delay_seconds)


def replace_previous_text(count: int, replacement: str, prefer_paste: bool = False) -> None:
    select_previous_chars(count)
    if prefer_paste:
        paste_text_with_restore(replacement)
    else:
        type_unicode_text(replacement)


def foreground_window() -> int | None:
    _require_windows()
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    return int(hwnd) if hwnd else None


def focus_window(hwnd: int | None, delay_seconds: float = 0.08) -> None:
    _require_windows()
    if not hwnd:
        return
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    time.sleep(delay_seconds)


def cursor_position() -> tuple[int, int] | None:
    _require_windows()
    point = wintypes.POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        return int(point.x), int(point.y)
    return None
