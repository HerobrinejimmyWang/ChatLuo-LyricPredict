from lyricpredict.context import extract_lyric_context


def test_extracts_last_quoted_lyric_suffix():
    assert extract_lyric_context("评论文字。“幸福从非泡影，以笑容证明，") == "幸福从非泡影，以笑容证明，"


def test_extracts_colon_introduced_lyric_suffix():
    text = "那首歌的歌词可能直接看一下会更明显一点：念往昔 我急旋慢转你抚琴低吟，到如今 重唱此曲却已无你"

    assert extract_lyric_context(text) == "念往昔 我急旋慢转你抚琴低吟，到如今 重唱此曲却已无你"


def test_ignores_closed_earlier_quote_before_colon():
    text = "我喜欢这首歌，可能是因为我“重别离”吧。歌词是：念往昔 我急旋慢转你抚琴低吟"

    assert extract_lyric_context(text) == "念往昔 我急旋慢转你抚琴低吟"


def test_keeps_clean_context_unchanged():
    assert extract_lyric_context("将故事传颂吧，风携它远追") == "将故事传颂吧，风携它远追"
