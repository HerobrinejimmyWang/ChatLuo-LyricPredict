from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .separators import HARD_TERMINATORS, ends_with_separator, starts_with_separator

TERMINATORS = HARD_TERMINATORS


def cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def should_keep_training_line(line: str, tokenizer, include_non_chinese: bool, max_unk_ratio: float) -> bool:
    if not line:
        return False
    if not include_non_chinese and cjk_count(line) == 0:
        return False
    token_ids = tokenizer.encode(line, add_special_tokens=False)
    if not token_ids:
        return False
    if tokenizer.unk_token_id is not None:
        unk_ratio = sum(1 for token_id in token_ids if token_id == tokenizer.unk_token_id) / len(token_ids)
        if unk_ratio > max_unk_ratio:
            return False
    return True


def ensure_training_terminator(line: str) -> str:
    return line if line.endswith(TERMINATORS) else f"{line}。"


def join_training_lines(lines: list[str]) -> str:
    pieces: list[str] = []
    for line in lines:
        if not pieces:
            pieces.append(line)
            continue
        separator = "" if ends_with_separator(pieces[-1]) or starts_with_separator(line) else "，"
        pieces.append(f"{separator}{line}")
    text = "".join(pieces)
    return ensure_training_terminator(text)


class TextBlockDataset:
    def __init__(
        self,
        path: Path,
        tokenizer,
        block_size: int,
        limit_blocks: int | None = None,
        include_non_chinese: bool = False,
        max_unk_ratio: float = 0.2,
    ):
        import torch

        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not text.strip():
            raise ValueError(f"No training text found in {path}. Run prepare first.")
        kept_text_lines: list[str] = []
        self.total_lines = 0
        self.kept_lines = 0
        self.skipped_lines = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            self.total_lines += 1
            if not should_keep_training_line(line, tokenizer, include_non_chinese, max_unk_ratio):
                self.skipped_lines += 1
                continue
            self.kept_lines += 1
            kept_text_lines.append(line)
        encoded = tokenizer.encode(join_training_lines(kept_text_lines), add_special_tokens=False)
        self.examples = []
        for index in range(0, max(1, len(encoded) - 1), block_size):
            block = encoded[index : index + block_size]
            if len(block) >= 8:
                tensor = torch.tensor(block, dtype=torch.long)
                self.examples.append({"input_ids": tensor, "labels": tensor.clone()})
            if limit_blocks is not None and len(self.examples) >= limit_blocks:
                break
        if not self.examples:
            raise ValueError(f"Not enough tokens to train from {path}.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features):
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            ids = feature["input_ids"]
            pad_length = max_length - len(ids)
            input_ids.append(torch.cat([ids, torch.full((pad_length,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_length, dtype=torch.long)]))
            labels.append(torch.cat([ids.clone(), torch.full((pad_length,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def configure_tokenizer_and_model(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.sep_token or tokenizer.cls_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = None
    model.config.bos_token_id = None
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def model_context_length(model) -> int:
    candidates = [
        getattr(model.config, "n_positions", None),
        getattr(model.config, "n_ctx", None),
        getattr(model.config, "max_position_embeddings", None),
    ]
    return min(value for value in candidates if isinstance(value, int) and value > 0)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and checkpoint_step(path) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the lyric model with LoRA when available.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional short-run override for smoke tests.")
    parser.add_argument("--output-dir", default=None, help="Optional model output directory override.")
    parser.add_argument("--limit-train-blocks", type=int, default=None, help="Optional dataset limit for smoke tests.")
    parser.add_argument("--limit-eval-blocks", type=int, default=None, help="Optional eval dataset limit for smoke tests.")
    parser.add_argument("--include-non-chinese", action="store_true", help="Keep lines without CJK characters.")
    parser.add_argument("--max-unk-ratio", type=float, default=0.2, help="Skip lines above this tokenizer [UNK] ratio.")
    parser.add_argument("--resume-lora", action="store_true", help="Continue training an existing LoRA adapter in output-dir.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint path, or 'auto' to use the largest output-dir/checkpoint-* step.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=None, help="Override config training epochs for this run.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config learning rate for this run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device train/eval batch size for this run.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps for this run.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config.model.base_model, use_safetensors=True, local_files_only=True)
    configure_tokenizer_and_model(tokenizer, model)

    block_size = min(config.training.block_size, model_context_length(model))
    if block_size != config.training.block_size:
        print(f"Training block_size clamped from {config.training.block_size} to model context length {block_size}.")

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if args.resume_lora and (output_dir / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(output_dir), is_trainable=True)
            print(f"Continuing trainable LoRA adapter from {output_dir}.")
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.training.lora_r,
                lora_alpha=config.training.lora_alpha,
                lora_dropout=config.training.lora_dropout,
                target_modules=["c_attn", "c_proj", "c_fc"],
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as exc:
        print(f"LoRA is unavailable; training full model instead: {exc}")

    train_dataset = TextBlockDataset(
        config.paths.processed_dir / "train.txt",
        tokenizer,
        block_size,
        limit_blocks=args.limit_train_blocks,
        include_non_chinese=args.include_non_chinese,
        max_unk_ratio=args.max_unk_ratio,
    )
    eval_path = config.paths.processed_dir / "valid.txt"
    eval_dataset = (
        TextBlockDataset(
            eval_path,
            tokenizer,
            block_size,
            limit_blocks=args.limit_eval_blocks,
            include_non_chinese=args.include_non_chinese,
            max_unk_ratio=args.max_unk_ratio,
        )
        if eval_path.exists() and eval_path.read_text(encoding="utf-8").strip()
        else None
    )
    print(
        f"Train lines kept/skipped: {train_dataset.kept_lines}/{train_dataset.skipped_lines}; "
        f"blocks: {len(train_dataset)}"
    )
    if eval_dataset is not None:
        print(
            f"Eval lines kept/skipped: {eval_dataset.kept_lines}/{eval_dataset.skipped_lines}; "
            f"blocks: {len(eval_dataset)}"
        )

    num_train_epochs = args.num_train_epochs if args.num_train_epochs is not None else config.training.num_train_epochs
    learning_rate = args.learning_rate if args.learning_rate is not None else config.training.learning_rate
    batch_size = args.batch_size if args.batch_size is not None else config.training.batch_size
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else config.training.gradient_accumulation_steps
    )
    resume_checkpoint: Path | str | None = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            resume_checkpoint = latest_checkpoint(output_dir)
            if resume_checkpoint is None:
                print(f"No checkpoint-* directory found in {output_dir}; starting without Trainer checkpoint resume.")
        else:
            resume_checkpoint = Path(args.resume_from_checkpoint)
            if not resume_checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[],
        use_cpu=config.model.device == "cpu",
    )
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "base_model": config.model.base_model,
                "block_size": block_size,
                "resume_lora": args.resume_lora,
                "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                "overrides": {
                    "num_train_epochs": args.num_train_epochs,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
