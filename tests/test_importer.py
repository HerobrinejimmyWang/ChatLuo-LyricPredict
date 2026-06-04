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
