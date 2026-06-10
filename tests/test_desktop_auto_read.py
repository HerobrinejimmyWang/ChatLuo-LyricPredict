from lyricpredict.desktop.auto_read import AutoReadCounter, auto_read_scope_allows, is_text_change_key


def test_text_change_key_filter_counts_text_and_edit_keys():
    assert is_text_change_key(ord("A"))
    assert is_text_change_key(0x20)
    assert is_text_change_key(0x08)
    assert is_text_change_key(0x0D)
    assert is_text_change_key(0xBA)


def test_text_change_key_filter_ignores_modifiers_and_shortcuts():
    assert not is_text_change_key(0x10)
    assert not is_text_change_key(ord("C"), ctrl_pressed=True)
    assert not is_text_change_key(ord("A"), alt_pressed=True)
    assert is_text_change_key(ord("V"), ctrl_pressed=True)


def test_auto_read_counter_threshold():
    counter = AutoReadCounter(threshold=3)

    assert not counter.record_change()
    assert not counter.record_change()
    assert counter.record_change()
    assert counter.count == 0


def test_auto_read_scope_allows_used_or_all_windows():
    assert auto_read_scope_allows("all-windows", 123, set())
    assert auto_read_scope_allows("used-windows", 123, {123})
    assert not auto_read_scope_allows("used-windows", 123, {456})
    assert not auto_read_scope_allows("all-windows", None, {123})
