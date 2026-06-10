import json
from pathlib import Path

import yaml

from lyricpredict.desktop.model_registry import load_model_registry
from lyricpredict.training_pipeline import merge_data_dirs, run_training_pipeline


def write_base_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "raw_dir": "data/raw",
                    "processed_dir": "data/processed",
                    "model_dir": "models/default",
                    "web_dir": "web",
                },
                "model": {"base_model": "mock"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_merge_data_dirs_appends_without_duplicates():
    assert merge_data_dirs(["selfdata/data1"], ["selfdata/data2", "selfdata/data1"]) == [
        "selfdata/data1",
        "selfdata/data2",
    ]
    assert merge_data_dirs(["selfdata/data1"], ["selfdata/data2"], replace_data=True) == ["selfdata/data2"]


def test_training_pipeline_creates_profile_and_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "configs" / "default.yaml"
    registry_path = tmp_path / "configs" / "models.yaml"
    write_base_config(base_config)

    executed = []
    run = run_training_pipeline(
        data_dirs=["selfdata/data1"],
        model_name="Luo Local",
        registry_path=str(registry_path),
        base_config_path=str(base_config),
        run_transformer=False,
        run_ngram=True,
        run_calibrate=True,
        runner=executed.append,
    )

    registry = load_model_registry(registry_path)
    profile = registry.profile("luo_local")
    runtime_config = yaml.safe_load((tmp_path / run.profile.config_path).read_text(encoding="utf-8"))
    state = json.loads((tmp_path / run.profile.model_dir / "training_pipeline.json").read_text(encoding="utf-8"))

    assert profile is not None
    assert profile.data_dirs == ["selfdata/data1"]
    assert runtime_config["paths"]["model_dir"] == "models/luo_local"
    assert [step.name for step in run.steps] == ["prepare", "export_ngram", "calibrate"]
    assert "--source-dir" in run.steps[0].command
    assert "selfdata/data1" in run.steps[0].command
    assert state["status"] == "completed"
    assert [step.name for step in executed] == ["prepare", "export_ngram", "calibrate"]


def test_training_pipeline_appends_data_to_existing_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "configs" / "default.yaml"
    registry_path = tmp_path / "configs" / "models.yaml"
    write_base_config(base_config)
    run_training_pipeline(
        data_dirs=["selfdata/data1"],
        model_name="Luo Local",
        registry_path=str(registry_path),
        base_config_path=str(base_config),
        run_transformer=False,
        runner=lambda step: None,
    )

    run = run_training_pipeline(
        data_dirs=["selfdata/data2"],
        model_id="luo_local",
        registry_path=str(registry_path),
        base_config_path=str(base_config),
        run_transformer=False,
        runner=lambda step: None,
    )

    registry = load_model_registry(registry_path)
    profile = registry.profile("luo_local")
    assert profile is not None
    assert profile.data_dirs == ["selfdata/data1", "selfdata/data2"]
    assert run.steps[0].command.count("--source-dir") == 2
    assert "selfdata/data1" in run.steps[0].command
    assert "selfdata/data2" in run.steps[0].command


def test_training_pipeline_dry_run_does_not_write_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "configs" / "default.yaml"
    registry_path = tmp_path / "configs" / "models.yaml"
    write_base_config(base_config)

    run = run_training_pipeline(
        data_dirs=["selfdata/data1"],
        model_name="Preview",
        registry_path=str(registry_path),
        base_config_path=str(base_config),
        run_transformer=False,
        dry_run=True,
    )

    assert run.dry_run
    assert not registry_path.exists()
    assert not (tmp_path / run.profile.config_path).exists()
    assert not (tmp_path / run.profile.model_dir / "training_pipeline.json").exists()
