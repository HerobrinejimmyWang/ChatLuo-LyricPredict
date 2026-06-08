from lyricpredict.cleaner import clean_lyrics_text


def test_lrc_timestamps_and_metadata_are_removed():
    text = "[ti:Song]\n[ar:Singer]\n[00:01.20]第一句歌词\n[00:03.40][00:04.00]第二句 歌词\n\n"
    song = clean_lyrics_text(text, "sample.lrc")
    assert song.lines == ["第一句歌词", "第二句 歌词"]


def test_txt_blank_lines_are_removed_and_spaces_normalized():
    text = "  hello   world  \n\n下一句　歌词"
    song = clean_lyrics_text(text, "sample.txt")
    assert song.lines == ["hello world", "下一句 歌词"]


def test_lrc_credit_lines_are_removed_after_timestamp_strip():
    text = "\n".join(
        [
            "[00:00.000]作词 Lyricist : LLABB",
            "[00:01.000]作曲 Composer : 小野道ono",
            "[00:02.000]编曲 Arrange : 小野道ono",
            "[00:03.000]调声 Tuning : 周小蚕",
            "[00:04.000]吉他 Guitar : Darkness Ten",
            "[00:05.000]混音 Mixing : 徐天鸿@Studio21A",
            "[00:06.000]母带 Mastering : 徐天鸿@Studio21A",
            "[00:07.000]制作人 Producer : 小野道ono",
            "[00:08.000]真正的歌词留下来",
        ]
    )

    song = clean_lyrics_text(text, "credits.lrc")

    assert song.lines == ["真正的歌词留下来"]
