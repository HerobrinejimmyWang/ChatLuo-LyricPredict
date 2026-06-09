from lyricpredict.desktop.qt_app import _parse_hotkey


def test_parse_hotkey_supports_letter_key():
    assert _parse_hotkey("Ctrl+Alt+L") == (0x0001 | 0x0002, ord("L"))


def test_parse_hotkey_supports_named_space_key():
    assert _parse_hotkey("Alt+Space") == (0x0001, 0x20)
