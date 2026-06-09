from lyricpredict.desktop.settings import AppSettings, coerce_settings, load_app_settings, save_app_settings


def test_app_settings_defaults_and_frequency_mapping():
    settings = AppSettings()

    assert settings.mode == "auto"
    assert settings.strictness == "balanced"
    assert settings.read_change_threshold == 8
    assert settings.suggestion_position == "bottom-right"
    assert settings.suggestion_style == "plain"
    assert settings.active_model_id == "default"


def test_app_settings_coerce_invalid_values():
    settings = coerce_settings(
        {
            "mode": "retrieval",
            "strictness": "wild",
            "context_window": 99,
            "suggestion_position": "center",
            "suggestion_style": "sparkles",
        }
    )

    assert settings.mode == "auto"
    assert settings.strictness == "balanced"
    assert settings.context_window == 32
    assert settings.suggestion_position == "bottom-right"
    assert settings.suggestion_style == "plain"


def test_app_settings_context_window_uses_fixed_steps():
    assert coerce_settings({"context_window": 12}).context_window == 10
    assert coerce_settings({"context_window": 20}).context_window == 16
    assert coerce_settings({"context_window": 28}).context_window == 24


def test_app_settings_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "app.yaml"
    expected = AppSettings(
        enabled=False,
        mode="model-only",
        strictness="strict",
        context_window=16,
        suggestion_position="top-left",
        suggestion_style="luo",
        active_model_id="quotes",
    )

    save_app_settings(expected, path)
    actual = load_app_settings(path)

    assert actual.enabled is False
    assert actual.mode == "model-only"
    assert actual.strictness == "strict"
    assert actual.context_window == 16
    assert actual.suggestion_position == "top-left"
    assert actual.suggestion_style == "luo"
    assert actual.active_model_id == "quotes"


def test_app_settings_save_and_load_newline_separator(tmp_path):
    path = tmp_path / "app.yaml"
    expected = AppSettings(default_separator="\n")

    save_app_settings(expected, path)
    actual = load_app_settings(path)

    assert actual.default_separator == "\n"


def test_app_settings_accepts_known_hotkey_and_rejects_unknown():
    assert coerce_settings({"hotkey": "Ctrl+Alt+P"}).hotkey == "Ctrl+Alt+P"
    assert coerce_settings({"hotkey": "Ctrl+Alt+Q"}).hotkey == "Ctrl+Alt+L"
