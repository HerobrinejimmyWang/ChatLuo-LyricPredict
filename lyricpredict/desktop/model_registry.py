from __future__ import annotations

import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


MODEL_ID_RE = re.compile(r"[^a-z0-9_-]+")


@dataclass(frozen=True)
class ModelProfile:
    id: str
    name: str
    config_path: str
    model_dir: str
    raw_dir: str
    processed_dir: str
    data_dirs: list[str] = field(default_factory=list)
    task: str = "lyrics"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ModelRegistry:
    active_model: str = "default"
    models: list[ModelProfile] = field(default_factory=list)

    def profile(self, model_id: str | None = None) -> ModelProfile | None:
        target = model_id or self.active_model
        for profile in self.models:
            if profile.id == target:
                return profile
        return self.models[0] if self.models else None


def slugify_model_id(name: str) -> str:
    value = MODEL_ID_RE.sub("_", name.strip().lower()).strip("_-")
    return value or "model"


def _default_profile(default_config_path: str = "configs/default.yaml") -> ModelProfile:
    return ModelProfile(
        id="default",
        name="Default Lyrics",
        config_path=default_config_path,
        model_dir="models/default",
        raw_dir="data/raw",
        processed_dir="data/processed",
        data_dirs=["selfdata/selflyricdata"],
    )


def _profile_from_mapping(data: dict[str, Any]) -> ModelProfile:
    return ModelProfile(
        id=str(data["id"]),
        name=str(data.get("name") or data["id"]),
        config_path=str(data.get("config_path") or "configs/default.yaml"),
        model_dir=str(data.get("model_dir") or "models/default"),
        raw_dir=str(data.get("raw_dir") or "data/raw"),
        processed_dir=str(data.get("processed_dir") or "data/processed"),
        data_dirs=[str(item) for item in data.get("data_dirs", [])],
        task=str(data.get("task") or "lyrics"),
        created_at=float(data.get("created_at") or time.time()),
        updated_at=float(data.get("updated_at") or time.time()),
    )


def load_model_registry(path: str | Path = "configs/models.yaml") -> ModelRegistry:
    registry_path = Path(path)
    if not registry_path.exists():
        return ModelRegistry(active_model="default", models=[_default_profile()])
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    models = [_profile_from_mapping(item) for item in data.get("models", []) if isinstance(item, dict)]
    if not any(profile.id == "default" for profile in models):
        models.insert(0, _default_profile())
    active = str(data.get("active_model") or "default")
    if not any(profile.id == active for profile in models):
        active = models[0].id
    return ModelRegistry(active_model=active, models=models)


def save_model_registry(registry: ModelRegistry, path: str | Path = "configs/models.yaml") -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_model": registry.active_model,
        "models": [asdict(profile) for profile in registry.models],
    }
    registry_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def create_runtime_config(profile: ModelProfile, base_config_path: str | Path = "configs/default.yaml") -> None:
    config_path = Path(profile.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(base_config_path).resolve() == config_path.resolve() and config_path.exists():
        return
    with Path(base_config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("paths", {})
    data["paths"]["raw_dir"] = profile.raw_dir
    data["paths"]["processed_dir"] = profile.processed_dir
    data["paths"]["model_dir"] = profile.model_dir
    data["paths"].setdefault("web_dir", "web")
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def create_model_profile(name: str, registry: ModelRegistry, base_config_path: str = "configs/default.yaml") -> ModelProfile:
    base_id = slugify_model_id(name)
    existing = {profile.id for profile in registry.models}
    model_id = base_id
    suffix = 2
    while model_id in existing:
        model_id = f"{base_id}_{suffix}"
        suffix += 1
    profile = ModelProfile(
        id=model_id,
        name=name.strip() or model_id,
        config_path=f"configs/models/{model_id}.yaml",
        model_dir=f"models/{model_id}",
        raw_dir=f"data/models/{model_id}/raw",
        processed_dir=f"data/models/{model_id}/processed",
        data_dirs=[],
    )
    Path(profile.raw_dir).mkdir(parents=True, exist_ok=True)
    Path(profile.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(profile.model_dir).mkdir(parents=True, exist_ok=True)
    create_runtime_config(profile, base_config_path)
    return profile


def copy_files_to_profile(files: list[str], profile: ModelProfile) -> int:
    raw_dir = Path(profile.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for file_name in files:
        source = Path(file_name)
        if not source.exists() or not source.is_file():
            continue
        target = raw_dir / source.name
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            index = 2
            while target.exists():
                target = raw_dir / f"{stem}_{index}{suffix}"
                index += 1
        shutil.copy2(source, target)
        count += 1
    return count
