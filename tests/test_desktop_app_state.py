from lyricpredict.desktop.app_state import DesktopAppState, load_desktop_app_state, save_desktop_app_state


def test_desktop_app_state_defaults_when_missing(tmp_path):
    state = load_desktop_app_state(tmp_path / "missing.yaml")

    assert state.auto_read_used_window_signatures == set()


def test_desktop_app_state_roundtrip(tmp_path):
    path = tmp_path / "app_state.yaml"
    expected = DesktopAppState(auto_read_used_window_signatures={"a|b", "c|d"})

    save_desktop_app_state(expected, path)
    actual = load_desktop_app_state(path)

    assert actual.auto_read_used_window_signatures == expected.auto_read_used_window_signatures
