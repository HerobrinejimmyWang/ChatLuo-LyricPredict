from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .confidence import ConfidenceGate, ConfidenceResult, ConfidenceSettings, load_confidence_settings
from .config import AppConfig
from .context import extract_lyric_context
from .ngram_model import CharNGramModel
from .retrieval import LyricRetriever

TERMINATORS = (",", ".", "，", "。")
CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,，.。])")
SPACE_AFTER_PUNCT_RE = re.compile(r"([,，.。])\s+")


@dataclass(frozen=True)
class Prediction:
    text: str
    accepted: bool
    confidence: float
    reason: str


def cut_at_terminator(text: str) -> tuple[str, bool]:
    has_leading_terminator = text[:1] in TERMINATORS
    search_start = 1 if has_leading_terminator else 0
    positions = [text.find(mark, search_start) for mark in TERMINATORS if text.find(mark, search_start) >= 0]
    if not positions:
        return text.strip(), False
    end = min(positions) if has_leading_terminator else min(positions) + 1
    return text[:end].strip(), True


def token_count_for_text(token_pieces: list[str], target_text: str) -> int:
    built = ""
    for index, piece in enumerate(token_pieces, start=1):
        built += piece
        if len(built) >= len(target_text):
            return index
    return len(token_pieces)


def decode_generated_ids(tokenizer, token_ids: list[int]) -> str:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = decoded.replace("##", "")
    decoded = CJK_SPACE_RE.sub("", decoded)
    decoded = SPACE_BEFORE_PUNCT_RE.sub(r"\1", decoded)
    decoded = SPACE_AFTER_PUNCT_RE.sub(r"\1", decoded)
    return decoded


def token_count_for_decoded_text(tokenizer, token_ids: list[int], target_text: str) -> int:
    for index in range(1, len(token_ids) + 1):
        decoded = decode_generated_ids(tokenizer, token_ids[:index]).strip()
        if decoded.startswith(target_text) or len(decoded) >= len(target_text):
            return index
    return len(token_ids)


def configure_tokenizer_and_model(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.sep_token or tokenizer.cls_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = None
    model.config.bos_token_id = None
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


class LyricGenerator:
    def __init__(
        self,
        config: AppConfig,
        mode: str | None = None,
        model_fallback_after_retrieval: bool | None = None,
    ):
        self.config = config
        self.mode = (mode or config.inference.mode).replace("_", "-").lower()
        if self.mode not in {"auto", "model-only", "retrieval"}:
            raise ValueError(f"Unsupported inference mode: {mode}")
        self.model_fallback_after_retrieval = (
            config.inference.model_fallback_after_retrieval
            if model_fallback_after_retrieval is None
            else model_fallback_after_retrieval
        )
        defaults = ConfidenceSettings(
            threshold=config.confidence.threshold,
            min_token_probability=config.confidence.min_token_probability,
            max_repeat_ratio=config.confidence.max_repeat_ratio,
        )
        self.confidence_gate = ConfidenceGate(load_confidence_settings(config.paths.model_dir, defaults))
        root_dir = config.paths.processed_dir.parent.parent
        self.retriever = LyricRetriever(config.paths.processed_dir, extra_dirs=(root_dir / "selflyricdata",))
        self.ngram_model: CharNGramModel | None | bool = False
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
            base = AutoModelForCausalLM.from_pretrained(base_model_name, use_safetensors=True, local_files_only=True)
            configure_tokenizer_and_model(self.tokenizer, base)
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(source, use_safetensors=True)
        configure_tokenizer_and_model(self.tokenizer, self.model)

        device = self.config.model.device
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        self.model.to(device)
        self.model.eval()

    def load_ngram_model(self) -> CharNGramModel | None:
        if self.ngram_model is False:
            self.ngram_model = CharNGramModel.load(self.config.paths.model_dir / "ngram_model.json")
        return self.ngram_model

    def predict(self, context: str) -> Prediction:
        context = context.strip()
        if not context:
            return Prediction("", False, 0.0, "empty_context")
        lyric_context = extract_lyric_context(context)
        if self.mode in {"auto", "retrieval"}:
            retrieved = self.retriever.find_next_line(lyric_context)
            if retrieved is not None:
                return Prediction(retrieved.text, True, retrieved.confidence, retrieved.reason)
            if self.mode == "retrieval" or not self.model_fallback_after_retrieval:
                return Prediction("", False, 0.0, "no_retrieval_match")
        ngram_model = self.load_ngram_model()
        if ngram_model is not None:
            ngram_prediction = ngram_model.predict(lyric_context, max_chars=self.config.model.max_new_tokens)
            if ngram_prediction is not None:
                result = self.confidence_gate.evaluate(ngram_prediction.text, [ngram_prediction.confidence], True)
                if result.accepted:
                    return Prediction(ngram_prediction.text, True, ngram_prediction.confidence, ngram_prediction.reason)
            return Prediction("", False, 0.0, "no_model_match")
        self.load()
        best_rejection = Prediction("", False, 0.0, "no_attempt")
        attempts = max(1, self.config.model.generation_attempts)
        for _ in range(attempts):
            prediction = self._predict_loaded(lyric_context)
            if prediction.accepted:
                return prediction
            if prediction.confidence >= best_rejection.confidence:
                best_rejection = prediction
        return best_rejection

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
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=None,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_ids = output.sequences[0][input_length:].detach().cpu().tolist()
        generated_text = decode_generated_ids(self.tokenizer, new_ids)
        cut_text, ended = cut_at_terminator(generated_text)
        used_tokens = token_count_for_decoded_text(self.tokenizer, new_ids, cut_text) if ended else len(new_ids)

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
