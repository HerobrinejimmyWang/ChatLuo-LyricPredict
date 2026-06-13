from lyricpredict.cleaner import clean_lyrics_text, is_credit_line, strip_speaker_label


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


def test_colon_metadata_fields_are_removed():
    metadata_lines = [
        "\u8bcd\uff1aGoosefleshh\uff08\u80e1\u542f\u8f89\uff09/\u4e8e\u5c0f\u4fde/\u5c0f\u516d",
        "\u7248\u6743\u58f0\u660e\uff1a\u672a\u7ecf\u8457\u4f5c\u6743\u4eba\u4e66\u9762\u8bb8\u53ef",
        "\u90ae\u7bb1\uff1awangxiangzengfu@163.com",
        "SP : \u9891\u7387\u878d\u5408",
        "OP/SP\uff1a\u660c\u79be\u6587\u5316",
        "Composed by\uff1aAubrey Graham/Ozan Yildirim",
        "P - Line: 2016 Sure Recordings Culture Co., Ltd",
        "Violin 1: \u5468\u606d\u5e73",
        "\u5236\u4f5c Production: Kay Production",
        "\u827a\u4eba\u7ecf\u7eaa Manager: Luuv Label",
    ]

    assert all(is_credit_line(line) for line in metadata_lines)


def test_speaker_labels_are_stripped_without_dropping_lyric_content():
    assert strip_speaker_label("\u6d1b\uff1a\u6d3b\u4e0b\u53bb") == "\u6d3b\u4e0b\u53bb"
    assert strip_speaker_label("\u4e9a\u7ec6\u4e9a\uff1a") is None
    assert strip_speaker_label("Joysa/\u6c88\u2f8d\u866b:we don't talk anymore") == "we don't talk anymore"
    assert strip_speaker_label("Joysa:I see your monsters") == "I see your monsters"
    assert strip_speaker_label("J and Keyshia are related: racist") == "J and Keyshia are related: racist"

    text = "[00:00.000]\u6d1b\uff1a\u6d3b\u4e0b\u53bb\n[00:01.000]\u4e9a\u7ec6\u4e9a\uff1a"
    song = clean_lyrics_text(text, "speaker.lrc")

    assert song.lines == ["\u6d3b\u4e0b\u53bb"]


def test_negative_offset_lrc_metadata_is_removed():
    text = "[00:00.00-1] \u4f5c\u8bcd : Armand Rodriguez\n[00:02.00-1]real lyric line"
    song = clean_lyrics_text(text, "negative-offset.lrc")

    assert song.lines == ["real lyric line"]
