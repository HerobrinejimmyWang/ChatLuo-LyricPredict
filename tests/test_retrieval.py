from lyricpredict.retrieval import LyricRetriever


def write_song(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(f"[00:00.00]{line}" for line in lines), encoding="utf-8")
    return path


def test_retriever_returns_next_line_with_leading_comma(tmp_path):
    write_song(tmp_path, "song.lrc", ["将故事传颂吧", "风携它远追", "你脸颊热泪"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("将故事传颂吧，风携它远追")

    assert result is not None
    assert result.text == "，你脸颊热泪"


def test_retriever_does_not_duplicate_leading_comma(tmp_path):
    write_song(tmp_path, "song.lrc", ["歌声冲破夜幕响彻在新的天地", "机械的心律带动血肉的共鸣"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("歌声冲破夜幕响彻在新的天地，")

    assert result is not None
    assert result.text == "机械的心律带动血肉的共鸣"


def test_retriever_rejects_ambiguous_next_lines(tmp_path):
    write_song(tmp_path, "song.lrc", ["一二三一二三", "左边", "一二三一二三", "右边"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    assert retriever.find_next_line("一二三一二三") is None


def test_retriever_allows_repeated_long_context_by_preferring_later_match(tmp_path):
    write_song(
        tmp_path,
        "song.lrc",
        [
            "上山岗 上山岗",
            "请听我 大声唱",
            "唱太阳 唱月亮",
            "唱时光 唱成长",
            "不孤单 不孤单",
            "上山岗 上山岗",
            "请听我 大声唱",
            "唱太阳 唱月亮",
            "唱时光 唱成长",
            "想起来 想起来 我们是否还一样",
        ],
    )
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("上山岗，上山岗，请听我，大声唱，唱太阳，唱月亮，唱时光，唱成长")

    assert result is not None
    assert result.text == "，想起来 想起来 我们是否还一样"


def test_retriever_allows_small_typo_in_long_context(tmp_path):
    write_song(tmp_path, "song.lrc", ["未来的你会光芒万丈", "而我也曾是你万分之一的光"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("未来的你会光茫万丈")

    assert result is not None
    assert result.text == "，而我也曾是你万分之一的光"
    assert result.corrected_context == "未来的你会光芒万丈"


def test_retriever_rejects_non_contiguous_suffix_match(tmp_path):
    write_song(tmp_path, "song.lrc", ["心事太多不愿只在梦里讲", "中间这一句必须出现", "如是思念辗转过山海无量", "错误命中"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    assert retriever.find_next_line("心事太多不愿只在梦里讲，如是思念辗转过山海无量") is None


def test_retriever_can_continue_inside_space_separated_line(tmp_path):
    write_song(tmp_path, "song.lrc", ["我明白 张开翅膀 需要有怎样的英勇", "我从来都不笨重 只是背着一个梦"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("我明白，张开翅膀需要有怎样的英勇，我从来都不笨重")

    assert result is not None
    assert result.text == "，只是背着一个梦"


def test_retriever_cuts_next_line_at_first_terminator(tmp_path):
    write_song(tmp_path, "song.lrc", ["念往昔，我急旋慢转你抚琴低吟", "到如今，重唱此曲却已无你"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("念往昔，我急旋慢转你抚琴低吟")

    assert result is not None
    assert result.text == "，到如今"


def test_retriever_continues_when_context_is_line_prefix(tmp_path):
    write_song(tmp_path, "song.lrc", ["还有我在你身边说我爱你啊"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("还有我在你身边说")

    assert result is not None
    assert result.text == "我爱你啊"


def test_retriever_allows_one_error_in_short_but_clear_context(tmp_path):
    write_song(tmp_path, "song.lrc", ["后来才懂得", "平常的事最值得纪念"])
    retriever = LyricRetriever(tmp_path / "processed", extra_dirs=(tmp_path,))

    result = retriever.find_next_line("到后来才懂得，")

    assert result is not None
    assert result.text == "平常的事最值得纪念"
