from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_MODES = {"auto", "model-only"}
VALID_STRICTNESS = {"strict", "balanced", "tolerant"}
VALID_FREQUENCIES = {"often", "sometimes", "seldom"}
VALID_CONTEXT_WINDOWS = (10, 16, 24, 32)
VALID_SUGGESTION_POSITIONS = {"bottom-right", "top-left"}
VALID_HOTKEYS = ("Ctrl+Alt+L", "Ctrl+Alt+P", "Ctrl+Shift+L", "Alt+Space")
VALID_SUGGESTION_STYLES = {"plain", "luo"}
READ_FREQUENCY_CHANGES = {"often": 4, "sometimes": 8, "seldom": 16}


@dataclass(frozen=True)
class AppSettings:
    enabled: bool = True
    mode: str = "auto"
    strictness: str = "balanced"
    context_window: int = 24
    read_frequency: str = "sometimes"
    correction: bool = True
    include_separator: bool = True
    default_separator: str = "，"
    suggestion_position: str = "bottom-right"
    suggestion_style: str = "plain"
    hotkey: str = "Ctrl+Alt+L"
    suggestion_timeout_seconds: float = 8.0
    lyric_config_path: str = "configs/default.yaml"
    model_registry_path: str = "configs/models.yaml"
    active_model_id: str = "default"
    window_x: int | None = None
    window_y: int | None = None

    @property
    def read_change_threshold(self) -> int:
        return READ_FREQUENCY_CHANGES.get(self.read_frequency, READ_FREQUENCY_CHANGES["sometimes"])


def _clamp_context_window(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = AppSettings.context_window
    return min(VALID_CONTEXT_WINDOWS, key=lambda candidate: abs(candidate - number))


def coerce_settings(data: dict[str, Any] | None = None) -> AppSettings:
    data = data or {}
    mode = str(data.get("mode", AppSettings.mode)).replace("_", "-").lower()
    strictness = str(data.get("strictness", AppSettings.strictness)).lower()
    read_frequency = str(data.get("read_frequency", AppSettings.read_frequency)).lower()
    suggestion_position = str(data.get("suggestion_position", AppSettings.suggestion_position)).replace("_", "-").lower()
    suggestion_style = str(data.get("suggestion_style", AppSettings.suggestion_style)).lower()
    return AppSettings(
        enabled=bool(data.get("enabled", AppSettings.enabled)),
        mode=mode if mode in VALID_MODES else AppSettings.mode,
        strictness=strictness if strictness in VALID_STRICTNESS else AppSettings.strictness,
        context_window=_clamp_context_window(data.get("context_window", AppSettings.context_window)),
        read_frequency=read_frequency if read_frequency in VALID_FREQUENCIES else AppSettings.read_frequency,
        correction=bool(data.get("correction", AppSettings.correction)),
        include_separator=bool(data.get("include_separator", AppSettings.include_separator)),
        default_separator=str(data.get("default_separator", AppSettings.default_separator) or AppSettings.default_separator),
        suggestion_position=(
            suggestion_position if suggestion_position in VALID_SUGGESTION_POSITIONS else AppSettings.suggestion_position
        ),
        suggestion_style=suggestion_style if suggestion_style in VALID_SUGGESTION_STYLES else AppSettings.suggestion_style,
        hotkey=str(data.get("hotkey", AppSettings.hotkey) or AppSettings.hotkey)
        if str(data.get("hotkey", AppSettings.hotkey) or AppSettings.hotkey) in VALID_HOTKEYS
        else AppSettings.hotkey,
        suggestion_timeout_seconds=float(data.get("suggestion_timeout_seconds", AppSettings.suggestion_timeout_seconds)),
        lyric_config_path=str(data.get("lyric_config_path", AppSettings.lyric_config_path) or AppSettings.lyric_config_path),
        model_registry_path=str(data.get("model_registry_path", AppSettings.model_registry_path) or AppSettings.model_registry_path),
        active_model_id=str(data.get("active_model_id", AppSettings.active_model_id) or AppSettings.active_model_id),
        window_x=data.get("window_x"),
        window_y=data.get("window_y"),
    )


def load_app_settings(path: str | Path = "configs/app.yaml") -> AppSettings:
    config_path = Path(path)
    if not config_path.exists():
        return AppSettings()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Desktop app settings must be a mapping: {config_path}")
    return coerce_settings(data)


def save_app_settings(settings: AppSettings, path: str | Path = "configs/app.yaml") -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(settings), handle, allow_unicode=True, sort_keys=False)
