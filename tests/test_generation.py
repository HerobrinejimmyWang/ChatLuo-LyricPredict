from lyricpredict.generation import cut_at_terminator, normalize_prediction_boundary, token_count_for_text


def test_cut_at_chinese_terminator_keeps_mark():
    assert cut_at_terminator("月光落下来，后面不要") == ("月光落下来，", True)


def test_cut_at_english_terminator_keeps_mark():
    assert cut_at_terminator("hello world. more") == ("hello world.", True)


def test_cut_allows_leading_comma_before_next_terminator():
    assert cut_at_terminator("，你脸颊热泪，后面不要") == ("，你脸颊热泪", True)


def test_cut_without_terminator_marks_unfinished():
    assert cut_at_terminator("没有结束符") == ("没有结束符", False)


def test_token_count_for_cut_text():
    assert token_count_for_text(["月", "光", "，", "后"], "月光，") == 3


def test_normalize_prediction_boundary_removes_duplicate_separator():
    assert normalize_prediction_boundary("未来的你会光芒万丈，", "，而我也曾是你万分之一的光") == "而我也曾是你万分之一的光"


def test_normalize_prediction_boundary_collapses_repeated_leading_separator():
    assert normalize_prediction_boundary("未来的你会光芒万丈", "，，而我也曾是你万分之一的光") == "，而我也曾是你万分之一的光"
