from lyricpredict.desktop.windows_io import VK_RETURN, _text_inputs


def test_text_inputs_turns_newline_into_return_key():
    inputs = _text_inputs("a\nb")

    assert inputs[2].union.ki.wVk == VK_RETURN
    assert inputs[3].union.ki.wVk == VK_RETURN
    assert inputs[2].union.ki.wScan == 0


def test_text_inputs_treats_crlf_as_one_return_key():
    inputs = _text_inputs("a\r\nb")

    return_keys = [item for item in inputs if item.union.ki.wVk == VK_RETURN]
    assert len(return_keys) == 2
