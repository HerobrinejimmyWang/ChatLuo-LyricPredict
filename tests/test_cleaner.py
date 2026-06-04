from lyricpredict.cleaner import clean_lyrics_text


def test_lrc_timestamps_and_metadata_are_removed():
    text = "[ti:Song]\n[ar:Singer]\n[00:01.20]第一句歌词\n[00:03.40][00:04.00]第二句 歌词\n\n"
    song = clean_lyrics_text(text, "sample.lrc")
    assert song.lines == ["第一句歌词", "第二句 歌词"]


def test_txt_blank_lines_are_removed_and_spaces_normalized():
    text = "  hello   world  \n\n下一句　歌词"
    song = clean_lyrics_text(text, "sample.txt")
    assert song.lines == ["hello world", "下一句 歌词"]
