from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lyricpredict.config import load_config
from lyricpredict.generation import (
    LyricGenerator,
    cut_at_terminator,
    decode_generated_ids,
    normalize_prediction_boundary,
    token_count_for_decoded_text,
)


def _clean_cell(value: str) -> str:
    return value.strip().strip("`").strip()


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell.strip()) <= {"-", ":"} for cell in cells if cell.strip())


def load_samples(path: Path, max_cases: int | None = None) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            samples = [str(item.get("input", item)) if isinstance(item, dict) else str(item) for item in data]
        elif isinstance(data, dict):
            samples = [str(item.get("input", item)) for item in data.get("samples", [])]
        else:
            samples = []
    else:
        samples = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("|") and line.endswith("|"):
                cells = [_clean_cell(cell) for cell in line.strip("|").split("|")]
                if _is_separator_row(cells) or not cells or cells[0].lower() == "no":
                    continue
                if cells[0].isdigit() and len(cells) >= 5:
                    input_index = 3 if cells[1].startswith("A-") and len(cells) > 3 else 2
                    samples.append(cells[input_index])
                continue
            if "|" not in line:
                samples.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        sample = sample.strip()
        if not sample or sample in seen:
            continue
        deduped.append(sample)
        seen.add(sample)
        if max_cases is not None and len(deduped) >= max_cases:
            break
    return deduped


def probe_transformer(generator: LyricGenerator, context: str, strictness: str) -> dict[str, Any]:
    import torch

    generator.load()
    assert generator.model is not None
    assert generator.tokenizer is not None

    encoded = generator.tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=generator.config.model.max_input_tokens,
    )
    device = next(generator.model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]

    with torch.no_grad():
        output = generator.model.generate(
            **encoded,
            max_new_tokens=generator.config.model.max_new_tokens,
            do_sample=True,
            temperature=generator.config.model.temperature,
            top_p=generator.config.model.top_p,
            repetition_penalty=generator.config.model.repetition_penalty,
            no_repeat_ngram_size=4,
            pad_token_id=generator.tokenizer.pad_token_id,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
        )

    new_ids = output.sequences[0][input_length:].detach().cpu().tolist()
    raw_text = decode_generated_ids(generator.tokenizer, new_ids)
    cut_text, ended = cut_at_terminator(raw_text)
    used_tokens = token_count_for_decoded_text(generator.tokenizer, new_ids, cut_text) if ended else len(new_ids)
    probabilities: list[float] = []
    for step, token_id in enumerate(new_ids[:used_tokens]):
        if step >= len(output.scores):
            break
        probs = torch.softmax(output.scores[step][0].detach().cpu(), dim=-1)
        probabilities.append(float(probs[token_id]))

    gate = generator._confidence_gate(generator._policy(strictness), "transformer")
    result = gate.evaluate(cut_text, probabilities, ended)
    final_text = normalize_prediction_boundary(context, cut_text) if result.accepted else ""
    return {
        "input": context,
        "raw_generated_text": raw_text,
        "cut_text": cut_text,
        "ended": ended,
        "confidence": result.confidence,
        "accepted": result.accepted,
        "rejection_reason": "accepted" if result.accepted else result.reason,
        "final_text": final_text,
    }


def probe_prediction(generator: LyricGenerator, context: str, strictness: str) -> dict[str, Any]:
    prediction = generator.predict(context, strictness=strictness)
    return {
        "input": context,
        "raw_generated_text": None,
        "cut_text": prediction.text,
        "ended": bool(prediction.text),
        "confidence": prediction.confidence,
        "accepted": prediction.accepted,
        "rejection_reason": "accepted" if prediction.accepted else prediction.reason,
        "final_text": prediction.text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a LyricPredict model with fixed sample contexts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=("transformer", "model-only", "auto"), default="transformer")
    parser.add_argument("--samples", default="testcase.md")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--strictness", choices=("strict", "balanced", "tolerant"), default="tolerant")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    args = parser.parse_args()

    samples = load_samples(Path(args.samples), args.max_cases)
    if not samples:
        raise SystemExit(f"No probe samples found in {args.samples}.")

    config = load_config(args.config)
    generator = LyricGenerator(config, mode="model-only" if args.mode == "transformer" else args.mode)
    rows = []
    for sample in samples:
        row = (
            probe_transformer(generator, sample, args.strictness)
            if args.mode == "transformer"
            else probe_prediction(generator, sample, args.strictness)
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
