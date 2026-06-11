from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from .desktop.model_registry import (
    ModelProfile,
    ModelRegistry,
    create_model_profile,
    create_runtime_config,
    load_model_registry,
    save_model_registry,
    slugify_model_id,
)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]


@dataclass(frozen=True)
class PipelineRun:
    profile: ModelProfile
    data_dirs: list[str]
    steps: list[PipelineStep]
    dry_run: bool = False
    options: dict | None = None


CommandRunner = Callable[[PipelineStep], None]


def _normalize_data_dir(value: str) -> str:
    return Path(value).as_posix()


def merge_data_dirs(existing: list[str], incoming: list[str], replace_data: bool = False) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in ([] if replace_data else existing) + incoming:
        normalized = _normalize_data_dir(value)
        if normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _replace_profile(registry: ModelRegistry, profile: ModelProfile) -> ModelRegistry:
    models = [profile if item.id == profile.id else item for item in registry.models]
    if not any(item.id == profile.id for item in models):
        models.append(profile)
    return ModelRegistry(active_model=profile.id, models=models)


def get_or_create_profile(
    model_id: str | None,
    model_name: str | None,
    registry: ModelRegistry,
    base_config_path: str,
    create_files: bool = True,
) -> ModelProfile:
    if model_id:
        profile = registry.profile(model_id)
        if profile is not None and profile.id == model_id:
            return profile
        return create_model_profile(model_name or model_id, registry, base_config_path) if create_files else preview_model_profile(model_name or model_id, registry)
    if model_name:
        return create_model_profile(model_name, registry, base_config_path) if create_files else preview_model_profile(model_name, registry)
    profile = registry.profile()
    if profile is None:
        raise ValueError("No active model profile found; pass --model-name to create one.")
    return profile


def preview_model_profile(name: str, registry: ModelRegistry) -> ModelProfile:
    base_id = slugify_model_id(name)
    existing = {profile.id for profile in registry.models}
    model_id = base_id
    suffix = 2
    while model_id in existing:
        model_id = f"{base_id}_{suffix}"
        suffix += 1
    return ModelProfile(
        id=model_id,
        name=name.strip() or model_id,
        config_path=f"configs/models/{model_id}.yaml",
        model_dir=f"models/{model_id}",
        raw_dir=f"data/models/{model_id}/raw",
        processed_dir=f"data/models/{model_id}/processed",
        data_dirs=[],
    )


def build_pipeline_steps(
    config_path: str,
    model_dir: str,
    data_dirs: list[str],
    run_transformer: bool = True,
    run_ngram: bool = True,
    run_calibrate: bool = True,
    ngram_order: int = 32,
    min_context: int = 8,
    max_steps: int | None = None,
    limit_train_blocks: int | None = None,
    limit_eval_blocks: int | None = None,
    resume_lora: bool = False,
    resume_from_checkpoint: str | None = None,
    num_train_epochs: float | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> list[PipelineStep]:
    prepare = [sys.executable, "-m", "lyricpredict.prepare", "--config", config_path]
    for data_dir in data_dirs:
        prepare.extend(["--source-dir", data_dir])
    steps = [PipelineStep("prepare", prepare)]

    if run_transformer:
        train = [
            sys.executable,
            "-m",
            "lyricpredict.train",
            "--config",
            config_path,
            "--output-dir",
            model_dir,
        ]
        if max_steps is not None:
            train.extend(["--max-steps", str(max_steps)])
        if limit_train_blocks is not None:
            train.extend(["--limit-train-blocks", str(limit_train_blocks)])
        if limit_eval_blocks is not None:
            train.extend(["--limit-eval-blocks", str(limit_eval_blocks)])
        if resume_lora:
            train.append("--resume-lora")
        if resume_from_checkpoint is not None:
            train.extend(["--resume-from-checkpoint", resume_from_checkpoint])
        if num_train_epochs is not None:
            train.extend(["--num-train-epochs", str(num_train_epochs)])
        if learning_rate is not None:
            train.extend(["--learning-rate", str(learning_rate)])
        if batch_size is not None:
            train.extend(["--batch-size", str(batch_size)])
        if gradient_accumulation_steps is not None:
            train.extend(["--gradient-accumulation-steps", str(gradient_accumulation_steps)])
        steps.append(PipelineStep("train_transformer", train))

    if run_ngram:
        steps.append(
            PipelineStep(
                "export_ngram",
                [
                    sys.executable,
                    "-m",
                    "lyricpredict.ngram_model",
                    "--config",
                    config_path,
                    "--order",
                    str(ngram_order),
                    "--min-context",
                    str(min_context),
                ],
            )
        )

    if run_calibrate:
        steps.append(PipelineStep("calibrate", [sys.executable, "-m", "lyricpredict.calibrate", "--config", config_path]))
    return steps


def write_pipeline_state(run: PipelineRun, status: str = "configured") -> None:
    model_dir = Path(run.profile.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_at": time.time(),
        "profile": asdict(run.profile),
        "data_dirs": run.data_dirs,
        "options": run.options or {},
        "steps": [{"name": step.name, "command": step.command} for step in run.steps],
    }
    (model_dir / "training_pipeline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_runner(step: PipelineStep) -> None:
    print(f"[{step.name}] {' '.join(step.command)}", flush=True)
    subprocess.run(step.command, check=True)


def run_training_pipeline(
    *,
    data_dirs: list[str],
    model_id: str | None = None,
    model_name: str | None = None,
    registry_path: str = "configs/models.yaml",
    base_config_path: str = "configs/default.yaml",
    replace_data: bool = False,
    run_transformer: bool = True,
    run_ngram: bool = True,
    run_calibrate: bool = True,
    ngram_order: int = 32,
    min_context: int = 8,
    max_steps: int | None = None,
    limit_train_blocks: int | None = None,
    limit_eval_blocks: int | None = None,
    resume_lora: bool = False,
    resume_from_checkpoint: str | None = None,
    num_train_epochs: float | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    dry_run: bool = False,
    runner: CommandRunner = default_runner,
) -> PipelineRun:
    registry = load_model_registry(registry_path)
    profile = get_or_create_profile(model_id, model_name, registry, base_config_path, create_files=not dry_run)
    merged_data_dirs = merge_data_dirs(profile.data_dirs, data_dirs, replace_data=replace_data)
    if not merged_data_dirs:
        raise ValueError("No data directories configured; pass one or more --data-dir values.")

    profile = replace(profile, data_dirs=merged_data_dirs, updated_at=time.time())
    steps = build_pipeline_steps(
        profile.config_path,
        profile.model_dir,
        merged_data_dirs,
        run_transformer=run_transformer,
        run_ngram=run_ngram,
        run_calibrate=run_calibrate,
        ngram_order=ngram_order,
        min_context=min_context,
        max_steps=max_steps,
        limit_train_blocks=limit_train_blocks,
        limit_eval_blocks=limit_eval_blocks,
        resume_lora=resume_lora,
        resume_from_checkpoint=resume_from_checkpoint,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    options = {
        "run_transformer": run_transformer,
        "run_ngram": run_ngram,
        "run_calibrate": run_calibrate,
        "resume_lora": resume_lora,
        "resume_from_checkpoint": resume_from_checkpoint,
        "overrides": {
            "max_steps": max_steps,
            "limit_train_blocks": limit_train_blocks,
            "limit_eval_blocks": limit_eval_blocks,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
    }
    run = PipelineRun(profile=profile, data_dirs=merged_data_dirs, steps=steps, dry_run=dry_run, options=options)
    if dry_run:
        for step in steps:
            print(f"[dry-run:{step.name}] {' '.join(step.command)}")
        return run

    create_runtime_config(profile, base_config_path)
    registry = _replace_profile(registry, profile)
    save_model_registry(registry, registry_path)
    write_pipeline_state(run, status="started")

    for step in steps:
        runner(step)
    write_pipeline_state(run, status="completed")
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end LyricPredict backend training pipeline.")
    parser.add_argument("--data-dir", action="append", default=[], help="Lyric source directory. Repeat to combine datasets.")
    parser.add_argument("--model-id", default=None, help="Existing model id to update. Defaults to active model.")
    parser.add_argument("--model-name", default=None, help="Create a new model profile with this name when needed.")
    parser.add_argument("--registry", default="configs/models.yaml")
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--replace-data", action="store_true", help="Replace profile data_dirs instead of appending.")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip LoRA/Transformer fine-tuning.")
    parser.add_argument("--skip-ngram", action="store_true", help="Skip n-gram export.")
    parser.add_argument("--skip-calibrate", action="store_true", help="Skip confidence calibration.")
    parser.add_argument("--ngram-order", type=int, default=32)
    parser.add_argument("--min-context", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-train-blocks", type=int, default=None)
    parser.add_argument("--limit-eval-blocks", type=int, default=None)
    parser.add_argument("--resume-lora", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run = run_training_pipeline(
        data_dirs=args.data_dir,
        model_id=args.model_id,
        model_name=args.model_name,
        registry_path=args.registry,
        base_config_path=args.base_config,
        replace_data=args.replace_data,
        run_transformer=not args.skip_transformer,
        run_ngram=not args.skip_ngram,
        run_calibrate=not args.skip_calibrate,
        ngram_order=args.ngram_order,
        min_context=args.min_context,
        max_steps=args.max_steps,
        limit_train_blocks=args.limit_train_blocks,
        limit_eval_blocks=args.limit_eval_blocks,
        resume_lora=args.resume_lora,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "model_id": run.profile.id,
                "config_path": run.profile.config_path,
                "model_dir": run.profile.model_dir,
                "data_dirs": run.data_dirs,
                "steps": [step.name for step in run.steps],
                "options": run.options,
                "dry_run": run.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
