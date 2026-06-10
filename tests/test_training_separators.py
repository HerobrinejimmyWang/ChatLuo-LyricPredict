from lyricpredict.separators import ends_with_separator, starts_with_separator, strip_leading_separators
from lyricpredict.train import join_training_lines


def test_shared_separator_semantics_include_space_and_newline():
    assert ends_with_separator("上一句 ")
    assert ends_with_separator("上一句\n")
    assert starts_with_separator(" 下一句")
    assert strip_leading_separators(" \n，下一句") == "下一句"


def test_join_training_lines_does_not_add_comma_after_existing_separator():
    assert join_training_lines(["上一句 ", "下一句"]) == "上一句 下一句。"
    assert join_training_lines(["上一句\n", "下一句"]) == "上一句\n下一句。"
    assert join_training_lines(["上一句", "下一句"]) == "上一句，下一句。"
