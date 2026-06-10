from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DesktopAppState:
    auto_read_used_window_signatures: set[str] = field(default_factory=set)


def load_desktop_app_state(path: str | Path = "configs/app_state.yaml") -> DesktopAppState:
    state_path = Path(path)
    if not state_path.exists():
        return DesktopAppState()
    with state_path.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return DesktopAppState()
    signatures = {
        str(item)
        for item in data.get("auto_read_used_window_signatures", [])
        if isinstance(item, str) and item.strip()
    }
    return DesktopAppState(auto_read_used_window_signatures=signatures)


def save_desktop_app_state(state: DesktopAppState, path: str | Path = "configs/app_state.yaml") -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "auto_read_used_window_signatures": sorted(state.auto_read_used_window_signatures),
    }
    state_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
