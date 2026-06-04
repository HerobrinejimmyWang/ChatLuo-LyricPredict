from lyricpredict.generation import cut_at_terminator, token_count_for_text


def test_cut_at_chinese_terminator_keeps_mark():
    assert cut_at_terminator("月光落下来，后面不要") == ("月光落下来，", True)


def test_cut_at_english_terminator_keeps_mark():
    assert cut_at_terminator("hello world. more") == ("hello world.", True)


def test_cut_without_terminator_marks_unfinished():
    assert cut_at_terminator("没有结束符") == ("没有结束符", False)


def test_token_count_for_cut_text():
    assert token_count_for_text(["月", "光", "，", "后"], "月光，") == 3
