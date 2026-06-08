from lyricpredict.importer import prepare_dataset, write_uploaded_file


def test_prepare_dataset_from_txt_and_lrc(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    write_uploaded_file(raw, "a.txt", "第一句\n第二句".encode("utf-8"))
    write_uploaded_file(raw, "b.lrc", "[00:01.00]第三句\n[ar:x]".encode("utf-8"))
    stats = prepare_dataset(raw, processed, validation_ratio=0.34)
    assert stats.files == 2
    assert stats.lines == 3
    assert (processed / "songs.jsonl").exists()
    assert "第一句" in (processed / "train.txt").read_text(encoding="utf-8")


def test_prepare_dataset_deduplicates_identical_cleaned_songs(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    write_uploaded_file(raw, "a.lrc", "[00:01.00]同一首歌\n[00:02.00]同一句歌词".encode("utf-8"))
    write_uploaded_file(raw, "b.lrc", "[00:03.00]同一首歌\n[00:04.00]同一句歌词".encode("utf-8"))

    stats = prepare_dataset(raw, processed, validation_ratio=0.0)

    assert stats.files == 2
    assert stats.songs == 1
    assert stats.lines == 2
    assert (processed / "train.txt").read_text(encoding="utf-8").count("同一首歌") == 1
