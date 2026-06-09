from __future__ import annotations

import yaml

from lyricpredict.desktop.model_registry import (
    ModelRegistry,
    copy_files_to_profile,
    create_model_profile,
    load_model_registry,
    save_model_registry,
    slugify_model_id,
)


def test_load_model_registry_returns_default_when_missing(tmp_path):
    registry = load_model_registry(tmp_path / "models.yaml")

    assert registry.active_model == "default"
    assert registry.profile("default") is not None
    assert registry.profile("default").model_dir == "models/default"


def test_slugify_model_id_keeps_simple_ascii_name():
    assert slugify_model_id("My Lyrics Model") == "my_lyrics_model"
    assert slugify_model_id("  ") == "model"


def test_create_model_profile_creates_isolated_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "configs" / "default.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "mock"},
                "paths": {
                    "raw_dir": "data/raw",
                    "processed_dir": "data/processed",
                    "model_dir": "models/default",
                    "web_dir": "web",
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    registry = load_model_registry(tmp_path / "configs" / "models.yaml")

    profile = create_model_profile("Famous Quotes", registry, str(base_config))

    assert profile.id == "famous_quotes"
    assert (tmp_path / profile.raw_dir).is_dir()
    assert (tmp_path / profile.processed_dir).is_dir()
    assert (tmp_path / profile.model_dir).is_dir()
    runtime_config = yaml.safe_load((tmp_path / profile.config_path).read_text(encoding="utf-8"))
    assert runtime_config["paths"]["raw_dir"] == profile.raw_dir
    assert runtime_config["paths"]["processed_dir"] == profile.processed_dir
    assert runtime_config["paths"]["model_dir"] == profile.model_dir


def test_save_and_load_model_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "configs" / "default.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text("paths: {}\n", encoding="utf-8")
    registry = load_model_registry(tmp_path / "configs" / "models.yaml")
    profile = create_model_profile("Lyrics B", registry, str(base_config))
    updated = ModelRegistry(active_model=profile.id, models=[*registry.models, profile])
    registry_path = tmp_path / "configs" / "models.yaml"

    save_model_registry(updated, registry_path)
    actual = load_model_registry(registry_path)

    assert actual.active_model == profile.id
    assert actual.profile(profile.id).name == "Lyrics B"


def test_copy_files_to_profile_renames_duplicate_imports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "configs" / "default.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text("paths: {}\n", encoding="utf-8")
    registry = load_model_registry(tmp_path / "configs" / "models.yaml")
    profile = create_model_profile("Dup Test", registry, str(base_config))
    source = tmp_path / "song.txt"
    source.write_text("hello", encoding="utf-8")

    assert copy_files_to_profile([str(source), str(source)], profile) == 2

    raw_dir = tmp_path / profile.raw_dir
    assert (raw_dir / "song.txt").read_text(encoding="utf-8") == "hello"
    assert (raw_dir / "song_2.txt").read_text(encoding="utf-8") == "hello"
