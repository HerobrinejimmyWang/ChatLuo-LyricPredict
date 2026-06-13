from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lyricpredict.config import AppConfig, load_config
from lyricpredict.matching_model import (
    build_candidate_library,
    make_predictor,
    train_bigru_ranker,
)
from lyricpredict.retrieval import _load_processed_songs
from scripts.evaluate_testresult import (
    EvalCase,
    build_recall_cases,
    build_situation_cases,
    exact_match,
    is_wrong_output,
    sample_cases,
)


ALLOWED_MODELS = {
    "legacy-ngram-generator",
    "legacy-ngram",
    "retrieval",
    "char-match",
    "char-bigru",
    "token-bigru",
}

ALLOWED_LINES = {"closed-library", "heldout-ranker"}


@dataclass
class MatchingEvalResult:
    correct: int
    wrong: int
    total: int
    latencies_ms: list[float]
    correction_full: int = 0
    correction_total: int = 0
    skipped: bool = False
    skip_reason: str = ""

    @property
    def abstain(self) -> int:
        return self.total - self.correct - self.wrong

    def display_accuracy(self) -> str:
        if self.skipped:
            return f"SKIPPED ({self.skip_reason})" if self.skip_reason else "SKIPPED"
        return _fraction(self.correct, self.total)

    def display_wrong(self) -> str:
        if self.skipped:
            return "SKIPPED"
        return _fraction(self.wrong, self.total)

    def display_abstain(self) -> str:
        if self.skipped:
            return "SKIPPED"
        return _fraction(self.abstain, self.total)

    def display_latency(self) -> str:
        if self.skipped:
            return "SKIPPED"
        if not self.latencies_ms:
            return "0.00 / 0.00 ms"
        return f"{statistics.mean(self.latencies_ms):.2f} / {statistics.median(self.latencies_ms):.2f} ms"

    def display_correction_full(self) -> str:
        if self.skipped:
            return "SKIPPED"
        return _fraction(self.correction_full, self.correction_total)


def _fraction(value: int, total: int) -> str:
    rate = value / total if total else 0.0
    return f"{value}/{total} ({rate:.1%})"


def parse_csv(raw: str) -> list[str]:
    return [part.strip().replace("_", "-").lower() for part in raw.split(",") if part.strip()]


def parse_models(raw: str) -> list[str]:
    models = parse_csv(raw)
    invalid = [model for model in models if model not in ALLOWED_MODELS]
    if invalid:
        raise SystemExit(f"Unsupported matching models: {', '.join(invalid)}")
    return ["legacy-ngram-generator" if model == "legacy-ngram" else model for model in models]


def parse_strictnesses(raw: str) -> list[str]:
    values = parse_csv(raw)
    allowed = {"strict", "balanced", "tolerant"}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise SystemExit(f"Unsupported strictness values: {', '.join(invalid)}")
    return values


def parse_lines(raw: str) -> list[str]:
    lines = parse_csv(raw)
    invalid = [line for line in lines if line not in ALLOWED_LINES]
    if invalid:
        raise SystemExit(f"Unsupported result lines: {', '.join(invalid)}")
    return lines


def parse_scenario_sample_sizes(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--scenario-sample-size must use Scenario=size format: {value}")
        scenario, raw_size = value.split("=", 1)
        scenario = scenario.strip()
        try:
            size = int(raw_size.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid sample size for {scenario}: {raw_size}") from exc
        if size <= 0:
            raise SystemExit(f"Scenario sample size must be positive: {value}")
        result[scenario] = size
    return result


def sample_ratio_cases(cases_by_scenario: dict[str, list[EvalCase]], ratio: float, seed: int) -> dict[str, list[EvalCase]]:
    if ratio <= 0 or ratio > 1:
        raise SystemExit("--sample-ratio must be in the range (0, 1].")
    sampled: dict[str, list[EvalCase]] = {}
    for offset, (name, cases) in enumerate(cases_by_scenario.items()):
        size = max(1, math.ceil(len(cases) * ratio)) if cases else 0
        sampled[name] = sample_cases(cases, size, seed + offset)
    return sampled


def sample_total_cases(cases_by_scenario: dict[str, list[EvalCase]], total_size: int, seed: int) -> dict[str, list[EvalCase]]:
    import random

    if total_size <= 0:
        return cases_by_scenario
    non_empty = {name: list(cases) for name, cases in cases_by_scenario.items() if cases}
    if not non_empty:
        return cases_by_scenario
    per_scenario = max(1, total_size // len(non_empty))
    remainder = max(0, total_size - per_scenario * len(non_empty))
    rng = random.Random(seed)
    sampled: dict[str, list[EvalCase]] = {}
    for offset, (name, cases) in enumerate(non_empty.items()):
        size = per_scenario + (1 if offset < remainder else 0)
        if len(cases) <= size:
            sampled[name] = list(cases)
            continue
        indexes = sorted(rng.sample(range(len(cases)), size))
        sampled[name] = [cases[index] for index in indexes]
    return sampled


def build_situation_cases_with_overrides(
    songs,
    sample_size: int,
    seed: int,
    scenario_sample_sizes: dict[str, int],
) -> dict[str, list[EvalCase]]:
    if not scenario_sample_sizes:
        return build_situation_cases(songs, sample_size, seed)
    max_size = max([sample_size, *scenario_sample_sizes.values()])
    cases = build_situation_cases(songs, max_size, seed)
    sampled: dict[str, list[EvalCase]] = {}
    for offset, (name, scenario_cases) in enumerate(cases.items()):
        size = scenario_sample_sizes.get(name, sample_size)
        sampled[name] = sample_cases(scenario_cases, size, seed + offset)
    return sampled


def build_cases(
    config: AppConfig,
    section: str,
    sample_size: int,
    seed: int,
    total_sample_size: int | None,
    sample_ratio: float | None,
    scenario_sample_sizes: dict[str, int],
) -> dict[str, list[EvalCase]]:
    songs = _load_processed_songs(config.paths.processed_dir)
    if not songs:
        raise SystemExit("No processed songs found. Run prepare first.")
    cases: dict[str, list[EvalCase]] = {}
    if section in {"recall", "all"}:
        recall_cases = build_recall_cases(songs, config, seed)
        if sample_ratio is not None:
            recall_cases = sample_ratio_cases(recall_cases, sample_ratio, seed)
        cases.update(recall_cases)
    if section in {"situations", "all"}:
        cases.update(build_situation_cases_with_overrides(songs, sample_size, seed, scenario_sample_sizes))
    if sample_ratio is not None and total_sample_size is not None:
        raise SystemExit("--sample-ratio and --total-sample-size cannot be used together.")
    if total_sample_size is not None:
        cases = sample_total_cases(cases, total_sample_size, seed)
    return cases


def parse_model_dirs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--model-dir must use model=path format: {value}")
        name, raw_path = value.split("=", 1)
        normalized = name.strip().replace("_", "-").lower()
        if normalized not in {"char-bigru", "token-bigru"}:
            raise SystemExit(f"--model-dir only supports char-bigru/token-bigru: {name}")
        result[normalized] = Path(raw_path.strip())
    return result


def maybe_train_missing(
    config: AppConfig,
    model_name: str,
    train_missing: bool,
    tokenizer: str | None,
    epochs: int,
    seed: int,
    model_dir: Path | None = None,
) -> str | None:
    if model_name not in {"char-bigru", "token-bigru"}:
        return None
    stem = model_name.replace("-", "_")
    artifact_dir = model_dir or config.paths.model_dir
    model_path = artifact_dir / f"matching_{stem}.pt"
    vocab_path = artifact_dir / f"matching_{stem}_vocab.json"
    config_path = artifact_dir / f"matching_{stem}_config.json"
    if model_path.exists() and vocab_path.exists() and config_path.exists():
        return None
    if not train_missing:
        return "missing_artifacts"
    train_bigru_ranker(
        config=config,
        variant=model_name,
        tokenizer_name=tokenizer,
        epochs=epochs,
        seed=seed,
        output_dir=artifact_dir,
    )
    return None


def evaluate_one(
    predictor,
    cases: list[EvalCase],
    strictness: str,
    line: str,
    model_name: str,
    scenario: str,
    failures: list[dict[str, Any]],
) -> MatchingEvalResult:
    correct = 0
    wrong = 0
    correction_full = 0
    correction_total = 0
    latencies: list[float] = []
    for case in cases:
        exclude_sources = {case.source} if line == "heldout-ranker" else None
        expects_correction = case.corrected_input is not None
        started = time.perf_counter()
        prediction = predictor.predict(
            case.input_text,
            strictness=strictness,
            exclude_sources=exclude_sources,
            correction=expects_correction,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        is_correct = exact_match(case.expected, prediction.text, prediction.accepted)
        is_wrong = is_wrong_output(case.expected, prediction.text, prediction.accepted)
        correct += int(is_correct)
        wrong += int(is_wrong)
        if expects_correction:
            correction_total += 1
            correction_full += int(is_correct and prediction.corrected_context == case.corrected_input)
        correction_mismatch = expects_correction and prediction.corrected_context != case.corrected_input
        if not is_correct or correction_mismatch:
            failures.append(
                {
                    "line": line,
                    "scenario": scenario,
                    "model": model_name,
                    "strictness": strictness,
                    "failure_type": "correction_context" if is_correct and correction_mismatch else "prediction",
                    "input": case.input_text,
                    "expected": case.expected,
                    "actual": prediction.text,
                    "accepted": prediction.accepted,
                    "confidence": prediction.confidence,
                    "reason": prediction.reason,
                    "expected_corrected_context": case.corrected_input,
                    "actual_corrected_context": prediction.corrected_context,
                    "source": case.source,
                    "latency_ms": elapsed_ms,
                }
            )
    return MatchingEvalResult(
        correct=correct,
        wrong=wrong,
        total=len(cases),
        latencies_ms=latencies,
        correction_full=correction_full,
        correction_total=correction_total,
    )


def markdown_table(
    rows: list[str],
    columns: list[str],
    results: dict[str, dict[str, MatchingEvalResult]],
    metric: str,
) -> str:
    header = f"|                 | {' | '.join(columns)} |"
    separator = f"|{'-' * 17}|{'|'.join('-' * (len(column) + 2) for column in columns)}|"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            result = results.get(row, {}).get(column, MatchingEvalResult(0, 0, 0, []))
            if metric == "wrong":
                values.append(result.display_wrong())
            elif metric == "abstain":
                values.append(result.display_abstain())
            elif metric == "latency":
                values.append(result.display_latency())
            else:
                values.append(result.display_accuracy())
        lines.append(f"| {row:<15} | {' | '.join(values)} |")
    return "\n".join(lines)


def combined_cell(result: MatchingEvalResult) -> str:
    if result.skipped:
        return f"SKIPPED ({result.skip_reason})" if result.skip_reason else "SKIPPED"
    return f"{result.correct} / {result.abstain} / {result.wrong}"


def combined_table(
    rows: list[str],
    columns: list[str],
    results: dict[str, dict[str, MatchingEvalResult]],
) -> str:
    header = f"| Scenario (N) | {' | '.join(columns)} |"
    separator = f"|{'-' * 14}|{'|'.join('-' * (len(column) + 2) for column in columns)}|"
    lines = [header, separator]
    for row in rows:
        row_results = results.get(row, {})
        total = next((result.total for result in row_results.values()), 0)
        values = [combined_cell(row_results.get(column, MatchingEvalResult(0, 0, 0, []))) for column in columns]
        lines.append(f"| {row} ({total}) | {' | '.join(values)} |")
    return "\n".join(lines)


def latency_table(
    rows: list[str],
    columns: list[str],
    results: dict[str, dict[str, MatchingEvalResult]],
) -> str:
    header = f"| Scenario | {' | '.join(columns)} |"
    separator = f"|{'-' * 10}|{'|'.join('-' * (len(column) + 2) for column in columns)}|"
    lines = [header, separator]
    for row in rows:
        row_results = results.get(row, {})
        values = [row_results.get(column, MatchingEvalResult(0, 0, 0, [])).display_latency() for column in columns]
        lines.append(f"| {row} | {' | '.join(values)} |")
    return "\n".join(lines)


def correction_full_table(
    rows: list[str],
    columns: list[str],
    results: dict[str, dict[str, MatchingEvalResult]],
) -> str:
    header = f"| Scenario | {' | '.join(columns)} |"
    separator = f"|{'-' * 10}|{'|'.join('-' * (len(column) + 2) for column in columns)}|"
    lines = [header, separator]
    for row in rows:
        row_results = results.get(row, {})
        values = [
            row_results.get(column, MatchingEvalResult(0, 0, 0, [])).display_correction_full()
            for column in columns
        ]
        lines.append(f"| {row} | {' | '.join(values)} |")
    return "\n".join(lines)


def total_cases(results: dict[str, dict[str, MatchingEvalResult]], rows: list[str]) -> int:
    total = 0
    for row in rows:
        row_results = results.get(row, {})
        total += next((result.total for result in row_results.values()), 0)
    return total


def render_report(
    results_by_line: dict[str, dict[str, dict[str, MatchingEvalResult]]],
    columns: list[str],
    recall_rows: list[str],
    situation_rows: list[str],
) -> str:
    parts = ["# Matching Model Evaluation", ""]
    for line, results in results_by_line.items():
        parts.extend([f"## {line}", ""])
        recall_results = {row: results[row] for row in recall_rows if row in results}
        if recall_results:
            recall_total = total_cases(recall_results, recall_rows)
            parts.extend(
                [
                    f"### Recall (N={recall_total})",
                    "",
                    "Cell format: correct / abstain / wrong.",
                    "",
                    combined_table(recall_rows, columns, recall_results),
                    "",
                    "### Recall Latency mean / median",
                    "",
                    latency_table(recall_rows, columns, recall_results),
                    "",
                ]
            )
        situation_results = {row: results[row] for row in situation_rows if row in results}
        if situation_results:
            situation_total = total_cases(situation_results, situation_rows)
            correction_rows = [row for row in ("Correction-one", "Correction-two") if row in situation_results]
            parts.extend(
                [
                    f"### Situations (N={situation_total})",
                    "",
                    "Cell format: correct / abstain / wrong.",
                    "",
                    combined_table(situation_rows, columns, situation_results),
                    "",
                    "### Correction Full",
                    "",
                    "Requires correct output and correct `corrected_context`.",
                    "",
                    correction_full_table(correction_rows, columns, situation_results),
                    "",
                    "### Situation Latency mean / median",
                    "",
                    latency_table(situation_rows, columns, situation_results),
                    "",
                ]
            )
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate experimental matching-style lyric models.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--models",
        default="legacy-ngram-generator,retrieval,char-match,char-bigru",
    )
    parser.add_argument("--section", choices=("recall", "situations", "all"), default="all")
    parser.add_argument("--strictnesses", default="strict,balanced,tolerant")
    parser.add_argument("--lines", default="closed-library")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--total-sample-size", type=int, default=None)
    parser.add_argument("--sample-ratio", type=float, default=None)
    parser.add_argument(
        "--scenario-sample-size",
        action="append",
        default=[],
        help="Override one situation sample size, e.g. 'Half-sentences=20'. Repeat as needed.",
    )
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--output", default="matching_results.generated.md")
    parser.add_argument("--failure-output", default="data/processed/matching_failures.json")
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--train-missing-epochs", type=int, default=4)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Artifact directory for a ranker, e.g. char-bigru=models/matching_char_bigru.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_names = parse_models(args.models)
    strictnesses = parse_strictnesses(args.strictnesses)
    lines = parse_lines(args.lines)
    model_dirs = parse_model_dirs(args.model_dir)
    scenario_sample_sizes = parse_scenario_sample_sizes(args.scenario_sample_size)
    cases_by_scenario = build_cases(
        config,
        args.section,
        args.sample_size,
        args.seed,
        args.total_sample_size,
        args.sample_ratio,
        scenario_sample_sizes,
    )
    songs = _load_processed_songs(config.paths.processed_dir)
    candidates = build_candidate_library(songs)
    columns = [f"{model}:{strictness}" for model in model_names for strictness in strictnesses]
    failures: list[dict[str, Any]] = []
    results_by_line: dict[str, dict[str, dict[str, MatchingEvalResult]]] = {
        line: {scenario: {} for scenario in cases_by_scenario}
        for line in lines
    }

    predictors = {}
    skipped: dict[str, str] = {}
    for model_name in model_names:
        skip_reason = maybe_train_missing(
            config,
            model_name,
            train_missing=args.train_missing,
            tokenizer=args.tokenizer,
            epochs=args.train_missing_epochs,
            seed=args.seed,
            model_dir=model_dirs.get(model_name),
        )
        if skip_reason:
            skipped[model_name] = skip_reason
            continue
        try:
            predictors[model_name] = make_predictor(model_name, config, candidates, model_dirs=model_dirs)
        except Exception as exc:
            skipped[model_name] = type(exc).__name__

    for line in lines:
        for scenario, cases in cases_by_scenario.items():
            for model_name in model_names:
                for strictness in strictnesses:
                    column = f"{model_name}:{strictness}"
                    if model_name in skipped:
                        results_by_line[line][scenario][column] = MatchingEvalResult(
                            correct=0,
                            wrong=0,
                            total=len(cases),
                            latencies_ms=[],
                            skipped=True,
                            skip_reason=skipped[model_name],
                        )
                        continue
                    results_by_line[line][scenario][column] = evaluate_one(
                        predictors[model_name],
                        cases,
                        strictness,
                        line,
                        model_name,
                        scenario,
                        failures,
                    )

    recall_rows = ["Multi-sentences", "Single sentence", "Complex context"]
    situation_rows = [
        "Half-sentences",
        "Symbols Outputs",
        "Correction-one",
        "Correction-two",
        "Mixed Context",
        "Out-of-library",
    ]
    report = render_report(results_by_line, columns, recall_rows, situation_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    failure_path = Path(args.failure_output)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report, end="")
    print(json.dumps({"failures": len(failures), "failure_output": str(failure_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
