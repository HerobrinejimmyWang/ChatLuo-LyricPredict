from lyricpredict.generation import (
    context_is_inside_clause,
    cut_at_terminator,
    normalize_prediction_boundary,
    token_count_for_text,
)


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


def test_context_inside_clause_after_existing_separator():
    assert context_is_inside_clause("我想要我想要你知道，不论")
    assert not context_is_inside_clause("未来的你会光芒万丈")
    assert not context_is_inside_clause("未来的你会光芒万丈，")
    assert not context_is_inside_clause("将故事传颂吧，风携它远追")
    assert not context_is_inside_clause("未来的你会光芒万丈，而我也曾是你万分之一的光，那么闪耀")
    assert not context_is_inside_clause("你应该忘记了吧，天气晴朗")


def test_normalize_prediction_boundary_removes_separator_inside_clause():
    assert normalize_prediction_boundary("我想要我想要你知道，不论", "，这世界多糟糕") == "这世界多糟糕"


def test_normalize_prediction_boundary_keeps_separator_after_complete_clause():
    assert normalize_prediction_boundary("将故事传颂吧，风携它远追", "，你脸颊热泪") == "，你脸颊热泪"
