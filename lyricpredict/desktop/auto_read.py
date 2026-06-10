from __future__ import annotations

from dataclasses import dataclass


VALID_AUTO_READ_SCOPES = {"used-windows", "all-windows"}

VK_BACK = 0x08
VK_RETURN = 0x0D
VK_SPACE = 0x20
VK_DELETE = 0x2E
VK_V = 0x56
MODIFIER_KEYS = {0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5}
TEXT_KEY_RANGES = ((0x30, 0x5A), (0x60, 0x6F), (0xBA, 0xC0), (0xDB, 0xDF))
TEXT_CONTROL_KEYS = {VK_BACK, VK_RETURN, VK_SPACE, VK_DELETE}


def is_text_change_key(vk_code: int, ctrl_pressed: bool = False, alt_pressed: bool = False) -> bool:
    if vk_code in MODIFIER_KEYS:
        return False
    if alt_pressed:
        return False
    if ctrl_pressed:
        return vk_code == VK_V
    if vk_code in TEXT_CONTROL_KEYS:
        return True
    return any(start <= vk_code <= end for start, end in TEXT_KEY_RANGES)


def auto_read_scope_allows(scope: str, hwnd: int | None, used_windows: set[int]) -> bool:
    if not hwnd:
        return False
    if scope == "all-windows":
        return True
    return hwnd in used_windows


@dataclass
class AutoReadCounter:
    threshold: int
    count: int = 0

    def record_change(self) -> bool:
        self.count += 1
        if self.count >= self.threshold:
            self.count = 0
            return True
        return False

    def reset(self) -> None:
        self.count = 0
