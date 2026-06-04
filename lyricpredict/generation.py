from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .confidence import ConfidenceGate, ConfidenceResult, ConfidenceSettings, load_confidence_settings
from .config import AppConfig

TERMINATORS = (",", ".", "，", "。")


@dataclass(frozen=True)
class Prediction:
    text: str
    accepted: bool
    confidence: float
    reason: str


def cut_at_terminator(text: str) -> tuple[str, bool]:
    positions = [text.find(mark) for mark in TERMINATORS if text.find(mark) >= 0]
    if not positions:
        return text.strip(), False
    end = min(positions) + 1
    return text[:end].strip(), True


def token_count_for_text(token_pieces: list[str], target_text: str) -> int:
    built = ""
    for index, piece in enumerate(token_pieces, start=1):
        built += piece
        if len(built) >= len(target_text):
            return index
    return len(token_pieces)


class LyricGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        defaults = ConfidenceSettings(
            threshold=config.confidence.threshold,
            min_token_probability=config.confidence.min_token_probability,
            max_repeat_ratio=config.confidence.max_repeat_ratio,
        )
        self.confidence_gate = ConfidenceGate(load_confidence_settings(config.paths.model_dir, defaults))
        self.tokenizer = None
        self.model = None

    def _model_source(self) -> str:
        model_dir = self.config.paths.model_dir
        if (model_dir / "config.json").exists() or (model_dir / "adapter_config.json").exists():
            return str(model_dir)
        return self.config.model.base_model

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = self._model_source()
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        adapter_path = Path(source)
        if (adapter_path / "adapter_config.json").exists():
            from peft import PeftModel

            base_model_name = self.config.model.base_model
            base = AutoModelForCausalLM.from_pretrained(base_model_name)
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(source)

        device = self.config.model.device
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        self.model.to(device)
        self.model.eval()

    def predict(self, context: str) -> Prediction:
        context = context.strip()
        if not context:
            return Prediction("", False, 0.0, "empty_context")
        self.load()
        return self._predict_loaded(context)

    def _predict_loaded(self, context: str) -> Prediction:
        import torch

        assert self.model is not None
        assert self.tokenizer is not None

        encoded = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.config.model.max_new_tokens,
                do_sample=True,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                repetition_penalty=self.config.model.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_ids = output.sequences[0][input_length:].detach().cpu().tolist()
        token_pieces = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in new_ids]
        generated_text = "".join(token_pieces)
        cut_text, ended = cut_at_terminator(generated_text)
        used_tokens = token_count_for_text(token_pieces, cut_text) if ended else len(new_ids)

        probabilities: list[float] = []
        for step, token_id in enumerate(new_ids[:used_tokens]):
            if step >= len(output.scores):
                break
            probs = torch.softmax(output.scores[step][0].detach().cpu(), dim=-1)
            probabilities.append(float(probs[token_id]))

        result: ConfidenceResult = self.confidence_gate.evaluate(cut_text, probabilities, ended)
        if not result.accepted:
            return Prediction("", False, result.confidence, result.reason)
        return Prediction(cut_text, True, result.confidence, result.reason)
