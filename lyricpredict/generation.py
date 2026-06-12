from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .confidence import ConfidenceGate, ConfidenceResult, ConfidenceSettings, load_confidence_profiles
from .config import AppConfig
from .context import extract_lyric_context
from .ngram_model import CharNGramModel
from .retrieval import LyricRetriever, RetrievalResult, _key

TERMINATORS = (",", ".", "，", "。")
CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,，.。])")
SPACE_AFTER_PUNCT_RE = re.compile(r"([,，.。])\s+")
REPEATED_TERMINATOR_RE = re.compile(r"([,.，。])\1+")
INCOMPLETE_CLAUSE_PREFIXES = (
    "不论",
    "无论",
    "不管",
    "就算",
    "即使",
    "哪怕",
    "如果",
    "只要",
    "直到",
    "当",
    "等到",
    "因为",
    "为了",
    "还有我",
)


@dataclass(frozen=True)
class Prediction:
    text: str
    accepted: bool
    confidence: float
    reason: str
    corrected_context: str | None = None


@dataclass(frozen=True)
class TransformerCandidate:
    text: str
    confidence: float
    reason: str
    raw_text: str = ""
    corrected_context: str | None = None


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    text: str
    score: float
    reason: str
    corrected_context: str | None = None


@dataclass(frozen=True)
class StrictnessPolicy:
    name: str
    threshold_delta: float = 0.0
    min_token_probability_scale: float = 1.0
    max_repeat_ratio_delta: float = 0.0
    allow_fuzzy: bool = True
    fuzzy_error_scale: float = 1.0
    allow_transformer_fallback: bool = False
    final_threshold: float = 0.62
    support_threshold: float = 0.35


STRICTNESS_POLICIES = {
    "strict": StrictnessPolicy(
        name="strict",
        threshold_delta=0.08,
        min_token_probability_scale=1.5,
        max_repeat_ratio_delta=-0.10,
        allow_fuzzy=True,
        fuzzy_error_scale=0.7,
        allow_transformer_fallback=False,
        final_threshold=0.80,
        support_threshold=0.65,
    ),
    "balanced": StrictnessPolicy(name="balanced", allow_transformer_fallback=True),
    "tolerant": StrictnessPolicy(
        name="tolerant",
        threshold_delta=-0.03,
        min_token_probability_scale=0.5,
        max_repeat_ratio_delta=0.15,
        allow_fuzzy=True,
        fuzzy_error_scale=1.5,
        allow_transformer_fallback=True,
        final_threshold=0.50,
        support_threshold=0.0,
    ),
}


def normalize_strictness(strictness: str | None, default: str = "balanced") -> str:
    value = (strictness or default).replace("_", "-").lower()
    return value if value in STRICTNESS_POLICIES else "balanced"


def cut_at_terminator(text: str) -> tuple[str, bool]:
    has_leading_terminator = text[:1] in TERMINATORS
    search_start = 1 if has_leading_terminator else 0
    positions = [text.find(mark, search_start) for mark in TERMINATORS if text.find(mark, search_start) >= 0]
    if not positions:
        return text.strip(), False
    end = min(positions) if has_leading_terminator else min(positions) + 1
    return text[:end].strip(), True


def context_is_inside_clause(context: str) -> bool:
    stripped = context.rstrip()
    if not stripped or stripped[-1:] in TERMINATORS:
        return False
    last_boundary = max(stripped.rfind(mark) for mark in TERMINATORS)
    if last_boundary < 0:
        return False
    suffix = stripped[last_boundary + 1 :].strip()
    suffix_key = re.sub(r"\s+", "", suffix)
    if not suffix_key:
        return False
    return any(suffix_key.startswith(prefix) for prefix in INCOMPLETE_CLAUSE_PREFIXES)


def normalize_prediction_boundary(context: str, text: str) -> str:
    text = REPEATED_TERMINATOR_RE.sub(r"\1", text.strip())
    if text[:1] in TERMINATORS and (context.rstrip()[-1:] in TERMINATORS or context_is_inside_clause(context)):
        return text.lstrip("".join(TERMINATORS)).strip()
    return text


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


def normalized_similarity(left: str, right: str) -> float:
    left_key = _key(left)
    right_key = _key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    previous = list(range(len(right_key) + 1))
    for left_index, left_char in enumerate(left_key, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right_key, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
            )
        previous = current
    distance = previous[-1]
    return max(0.0, 1.0 - distance / max(len(left_key), len(right_key)))


def boundary_score(context: str, text: str) -> float:
    if not text.strip():
        return 0.0
    if text[:1] in TERMINATORS and context.rstrip()[-1:] in TERMINATORS:
        return 0.35
    if text[:1] not in TERMINATORS and context.rstrip()[-1:] not in TERMINATORS and not context_is_inside_clause(context):
        return 0.65
    return 1.0


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
        self.confidence_profiles = load_confidence_profiles(config.paths.model_dir, defaults)
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

    def _policy(self, strictness: str | None = None) -> StrictnessPolicy:
        name = normalize_strictness(strictness, self.config.inference.strictness)
        return STRICTNESS_POLICIES[name]

    def _confidence_gate(self, policy: StrictnessPolicy, source: str) -> ConfidenceGate:
        profile = self.confidence_profiles.profile(source)
        settings = ConfidenceSettings(
            threshold=max(0.0, min(1.0, profile.threshold + policy.threshold_delta)),
            min_token_probability=max(0.0, profile.min_token_probability * policy.min_token_probability_scale),
            max_repeat_ratio=max(0.0, min(1.0, profile.max_repeat_ratio + policy.max_repeat_ratio_delta)),
        )
        return ConfidenceGate(settings)

    def _retrieval_hint_threshold(self, policy: StrictnessPolicy) -> float:
        profile = self.confidence_profiles.profile("retrieval")
        direct_threshold = max(0.0, min(1.0, profile.threshold + policy.threshold_delta))
        return max(0.35, direct_threshold - 0.35)

    def predict(self, context: str, strictness: str | None = None, correction: bool = False) -> Prediction:
        policy = self._policy(strictness)
        context = context.strip()
        if not context:
            return Prediction("", False, 0.0, "empty_context")
        lyric_context = extract_lyric_context(context)
        retrieval_hint: RetrievalResult | None = None
        if self.mode in {"auto", "retrieval"}:
            retrieved = self.retriever.find_next_line(lyric_context)
            if retrieved is not None:
                corrected_context = retrieved.corrected_context if correction else None
                boundary_context = corrected_context or lyric_context
                text = normalize_prediction_boundary(boundary_context, retrieved.text)
                result = self._confidence_gate(policy, "retrieval").evaluate(
                    text,
                    [retrieved.confidence],
                    True,
                )
                if result.accepted:
                    return Prediction(text, True, retrieved.confidence, retrieved.reason, corrected_context)
                if retrieved.confidence >= self._retrieval_hint_threshold(policy):
                    retrieval_hint = retrieved
                if self.mode == "retrieval" or not self.model_fallback_after_retrieval:
                    return Prediction("", False, result.confidence, result.reason, corrected_context)
            if self.mode == "retrieval" or not self.model_fallback_after_retrieval:
                return Prediction("", False, 0.0, "no_retrieval_match")

        candidates = self._generate_transformer_candidates(lyric_context, policy)
        decision = self._compare_candidates(lyric_context, candidates, retrieval_hint, policy, correction=correction)
        return Prediction(
            decision.text if decision.accepted else "",
            decision.accepted,
            decision.score,
            decision.reason,
            decision.corrected_context,
        )

    def _generate_transformer_candidates(
        self,
        context: str,
        policy: StrictnessPolicy,
    ) -> list[TransformerCandidate]:
        self.load()
        confidence_gate = self._confidence_gate(policy, "transformer")
        attempts = max(1, self.config.model.generation_attempts)
        candidates: list[TransformerCandidate] = []
        seen: set[str] = set()
        for _ in range(attempts):
            prediction = self._predict_loaded(context, confidence_gate)
            if not prediction.accepted or not prediction.text:
                continue
            identity = _key(prediction.text)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                TransformerCandidate(
                    text=prediction.text,
                    confidence=prediction.confidence,
                    reason=prediction.reason,
                    corrected_context=prediction.corrected_context,
                )
            )
        return candidates

    def _compare_candidates(
        self,
        context: str,
        candidates: list[TransformerCandidate],
        retrieval_hint: RetrievalResult | None,
        policy: StrictnessPolicy,
        correction: bool = False,
    ) -> CandidateDecision:
        if not candidates:
            return CandidateDecision(False, "", 0.0, "no_transformer_candidate")

        ngram_model = self.load_ngram_model()
        best = CandidateDecision(False, "", 0.0, "low_final_confidence")
        for candidate in candidates:
            text = normalize_prediction_boundary(context, candidate.text)
            ngram_score = 0.0
            ngram_reason = "ngram_unavailable"
            ngram_corrected_context: str | None = None
            if ngram_model is not None:
                verification = ngram_model.verify(
                    context,
                    text,
                    allow_fuzzy=policy.allow_fuzzy,
                    fuzzy_error_scale=policy.fuzzy_error_scale,
                )
                ngram_score = verification.score
                ngram_reason = verification.reason
                ngram_corrected_context = verification.corrected_context

            retrieval_score = normalized_similarity(retrieval_hint.text, text) if retrieval_hint is not None else 0.0
            boundary = boundary_score(context, text)
            final_score = (
                candidate.confidence * 0.45
                + ngram_score * 0.30
                + retrieval_score * 0.20
                + boundary * 0.05
            )
            support = max(ngram_score, retrieval_score)
            corrected_context = None
            if correction:
                corrected_context = (
                    retrieval_hint.corrected_context
                    if retrieval_hint is not None and retrieval_score >= 0.5
                    else ngram_corrected_context
                )
            reason = f"verified_transformer:{ngram_reason}"
            accepted = final_score >= policy.final_threshold and support >= policy.support_threshold
            decision = CandidateDecision(accepted, text if accepted else "", final_score, reason, corrected_context)
            if decision.accepted:
                return decision
            if final_score >= best.score:
                best = CandidateDecision(False, "", final_score, "low_final_confidence", corrected_context)
        return best

    def _predict_loaded(self, context: str, confidence_gate: ConfidenceGate) -> Prediction:
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

        result: ConfidenceResult = confidence_gate.evaluate(cut_text, probabilities, ended)
        if not result.accepted:
            return Prediction("", False, result.confidence, result.reason)
        return Prediction(normalize_prediction_boundary(context, cut_text), True, result.confidence, result.reason)
