import json

from lyricpredict.importer import prepare_dataset, prepare_dataset_from_sources, write_uploaded_file


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


def test_prepare_dataset_from_sources_combines_sources_and_writes_manifest(tmp_path):
    raw_a = tmp_path / "raw_a"
    raw_b = tmp_path / "raw_b"
    processed = tmp_path / "processed"
    write_uploaded_file(raw_a, "a.lrc", "[00:01.00]第一句\n[00:02.00]第二句".encode("utf-8"))
    write_uploaded_file(raw_b, "b.lrc", "[00:01.00]第三句\n[00:02.00]第四句".encode("utf-8"))

    stats = prepare_dataset_from_sources([raw_a, raw_b], processed, validation_ratio=0.0)

    manifest = json.loads((processed / "source_manifest.json").read_text(encoding="utf-8"))
    assert stats.files == 2
    assert stats.songs == 2
    assert stats.lines == 4
    assert manifest["stats"]["files"] == 2
    assert [source["files"] for source in manifest["sources"]] == [1, 1]
    assert "第一句" in (processed / "train.txt").read_text(encoding="utf-8")
    assert "第三句" in (processed / "train.txt").read_text(encoding="utf-8")


def test_prepare_dataset_filters_credit_metadata_variants(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    write_uploaded_file(
        raw,
        "credits.lrc",
        "\n".join(
            [
                "[00:01.00]第一句歌词",
                "[00:02.00]VOCALOID调校：某某",
                "[00:03.00]PV：某某",
                "[00:04.00]演唱 Vocal：洛天依",
                "[00:05.00]歌曲PV：av123",
                "[00:06.00]PV/封面设计：Ansa",
                "[00:07.00]第二句歌词",
            ]
        ).encode("utf-8"),
    )

    stats = prepare_dataset(raw, processed, validation_ratio=0.0)
    train_text = (processed / "train.txt").read_text(encoding="utf-8")

    assert stats.lines == 2
    assert "第一句歌词" in train_text
    assert "第二句歌词" in train_text
    assert "VOCALOID" not in train_text
    assert "PV" not in train_text
    assert "演唱" not in train_text
