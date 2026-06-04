from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


class TextBlockDataset:
    def __init__(self, path: Path, tokenizer, block_size: int):
        import torch

        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not text.strip():
            raise ValueError(f"No training text found in {path}. Run prepare first.")
        encoded = tokenizer(text, add_special_tokens=True)["input_ids"]
        self.examples = []
        for index in range(0, max(1, len(encoded) - 1), block_size):
            block = encoded[index : index + block_size]
            if len(block) >= 8:
                tensor = torch.tensor(block, dtype=torch.long)
                self.examples.append({"input_ids": tensor, "labels": tensor.clone()})
        if not self.examples:
            raise ValueError(f"Not enough tokens to train from {path}.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the lyric model with LoRA when available.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model.base_model)
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
    except Exception as exc:
        print(f"LoRA is unavailable; training full model instead: {exc}")

    train_dataset = TextBlockDataset(config.paths.processed_dir / "train.txt", tokenizer, config.training.block_size)
    eval_path = config.paths.processed_dir / "valid.txt"
    eval_dataset = TextBlockDataset(eval_path, tokenizer, config.training.block_size) if eval_path.exists() and eval_path.read_text(encoding="utf-8").strip() else None

    output_dir = config.paths.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[],
        no_cuda=config.model.device == "cpu",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(json.dumps({"base_model": config.model.base_model}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
