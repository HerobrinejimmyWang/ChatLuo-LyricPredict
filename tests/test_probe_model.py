from scripts.probe_model import load_samples


def test_probe_load_samples_reads_markdown_table(tmp_path):
    samples = tmp_path / "samples.md"
    samples.write_text(
        """
| NO | Group | Type | Input | Expected |
|----|-------|------|-------|----------|
| 1 | A-1 | single | hello lyric | next lyric |
| 2 | B-1 | noisy input | target | should output |
""",
        encoding="utf-8",
    )

    assert load_samples(samples) == ["hello lyric", "noisy input"]
