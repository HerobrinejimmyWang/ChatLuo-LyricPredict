from lyricpredict.train import latest_checkpoint


def test_latest_checkpoint_selects_largest_step(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "checkpoint-6").mkdir(parents=True)
    (model_dir / "checkpoint-18").mkdir()
    (model_dir / "checkpoint-12").mkdir()
    (model_dir / "checkpoint-final").mkdir()

    assert latest_checkpoint(model_dir) == model_dir / "checkpoint-18"


def test_latest_checkpoint_returns_none_when_missing(tmp_path):
    assert latest_checkpoint(tmp_path / "model") is None
