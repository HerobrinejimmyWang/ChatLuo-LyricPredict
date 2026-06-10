from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lyricpredict.cleaner import CleanedSong
from lyricpredict.config import AppConfig, load_config
from lyricpredict.generation import LyricGenerator
from lyricpredict.retrieval import TERMINATORS, _is_usable_line, _key, _load_processed_songs

TARGET_SOURCE_DIRS = ("selfdata/selflyricdata", "selfdata/selflyricdata2")
NOISE_SENTENCES = (
    "这里是一段普通说明文字，不属于任何歌词。",
    "用户正在回忆一首歌的氛围，下面才是需要续写的内容。",
    "这句话只是上下文干扰，请从最后出现的歌词继续。",
    "我先写一点和歌曲无关的评论，再接上真正的歌词。",
)
TYPO_REPLACEMENTS = (
    ("颂", "诵"),
    ("芒", "茫"),
    ("撼", "憾"),
    ("尘", "辰"),
    ("寰", "环"),
    ("晴", "青"),
    ("文", "闻"),
    ("依", "衣"),
    ("啊", "呀"),
    ("的", "地"),
    ("得", "的"),
    ("在", "再"),
)


@dataclass(frozen=True)
class EvalCase:
    scenario: str
    input_text: str
    expected: str
    source: str


@dataclass(frozen=True)
class EvalResult:
    correct: int
    wrong: int
    total: int
    skipped: bool = False

    @property
    def rejected(self) -> int:
        return self.total - self.correct - self.wrong

    def display_accuracy(self) -> str:
        if self.skipped:
            return "SKIPPED"
        rate = self.correct / self.total if self.total else 0.0
        return f"{self.correct}/{self.total} ({rate:.1%})"

    def display_wrong(self) -> str:
        if self.skipped:
            return "SKIPPED"
        rate = self.wrong / self.total if self.total else 0.0
        return f"{self.wrong}/{self.total} ({rate:.1%})"


def cut_next_text(text: str) -> str:
    positions = [text.find(mark) for mark in TERMINATORS if text.find(mark) >= 0]
    if not positions:
        return text.strip()
    return text[: min(positions)].strip()


def join_lyric_lines(lines: Iterable[str]) -> str:
    pieces: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if not pieces or pieces[-1].endswith(TERMINATORS) or line.startswith(TERMINATORS):
            pieces.append(line)
        else:
            pieces.append(f"，{line}")
    return "".join(pieces)


def format_expected(context: str, next_line: str) -> str:
    next_text = cut_next_text(next_line)
    if not next_text:
        return ""
    prefix = "" if context.rstrip().endswith(TERMINATORS) or next_text.startswith(TERMINATORS) else "，"
    return f"{prefix}{next_text}"


def exact_match(expected: str, actual: str, accepted: bool) -> bool:
    return accepted and actual == expected


def is_wrong_output(expected: str, actual: str, accepted: bool) -> bool:
    return accepted and actual != expected


def make_token_counter(config: AppConfig) -> Callable[[str], int]:
    model_dir = config.paths.model_dir
    source = str(model_dir) if (model_dir / "tokenizer.json").exists() else config.model.base_model
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    except Exception:
        return lambda text: len(text)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count


def usable_song_lines(song: CleanedSong) -> list[str]:
    return [line.strip() for line in song.lines if _is_usable_line(line)]


def adjacent_pairs(songs: list[CleanedSong]) -> list[tuple[CleanedSong, list[str], int]]:
    pairs: list[tuple[CleanedSong, list[str], int]] = []
    for song in songs:
        lines = usable_song_lines(song)
        for index in range(len(lines) - 1):
            expected = format_expected(lines[index], lines[index + 1])
            if expected:
                pairs.append((song, lines, index))
    return pairs


def build_single_sentence_cases(songs: list[CleanedSong]) -> list[EvalCase]:
    pairs = adjacent_pairs(songs)
    outputs_by_context: dict[str, set[str]] = defaultdict(set)
    for _, lines, index in pairs:
        outputs_by_context[_key(lines[index])].add(format_expected(lines[index], lines[index + 1]))

    cases: list[EvalCase] = []
    for song, lines, index in pairs:
        context = lines[index]
        if len(outputs_by_context[_key(context)]) > 1:
            continue
        cases.append(
            EvalCase(
                scenario="Single sentence",
                input_text=context,
                expected=format_expected(context, lines[index + 1]),
                source=song.source,
            )
        )
    return cases


def build_multi_sentence_cases(songs: list[CleanedSong], token_count: Callable[[str], int], min_tokens: int = 32) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for song, lines, index in adjacent_pairs(songs):
        start = index
        context = join_lyric_lines(lines[start : index + 1])
        while start > 0 and token_count(context) < min_tokens:
            start -= 1
            context = join_lyric_lines(lines[start : index + 1])
        if token_count(context) < min_tokens:
            continue
        cases.append(
            EvalCase(
                scenario="Multi-sentences",
                input_text=context,
                expected=format_expected(context, lines[index + 1]),
                source=song.source,
            )
        )
    return cases


def build_complex_context_cases(songs: list[CleanedSong], seed: int) -> list[EvalCase]:
    cases: list[EvalCase] = []
    noise = list(NOISE_SENTENCES)
    rng = random.Random(seed)
    rng.shuffle(noise)
    for offset, case in enumerate(build_single_sentence_cases(songs)):
        cases.append(
            EvalCase(
                scenario="Complex context",
                input_text=f"{noise[offset % len(noise)]}{case.input_text}",
                expected=case.expected,
                source=case.source,
            )
        )
    return cases


def split_clause(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,，.。]", line) if _is_usable_line(part)]


def build_half_sentence_pool(songs: list[CleanedSong]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for song in songs:
        for line in usable_song_lines(song):
            for part in split_clause(line):
                key = _key(part)
                if len(key) < 8:
                    continue
                midpoint = max(2, len(part) // 2)
                input_text = part[:midpoint].strip()
                expected = part[midpoint:].strip()
                if _is_usable_line(input_text) and _is_usable_line(expected):
                    cases.append(EvalCase("Half-sentences", input_text, expected, song.source))
    return cases


def build_symbol_pool(songs: list[CleanedSong]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for base in build_single_sentence_cases(songs):
        stripped = base.input_text.rstrip("".join(TERMINATORS))
        if not stripped:
            continue
        variants = (stripped, f"{stripped}，", f"{stripped},", f"{stripped}。")
        seen: set[str] = set()
        for variant in variants:
            if variant in seen:
                continue
            seen.add(variant)
            cases.append(
                EvalCase(
                    scenario="Symbols Outputs",
                    input_text=variant,
                    expected=format_expected(variant, base.expected.lstrip("".join(TERMINATORS))),
                    source=base.source,
                )
            )
    return cases


def apply_typos(text: str, count: int) -> str | None:
    changed = text
    used = 0
    for old, new in TYPO_REPLACEMENTS:
        if old in changed and new not in changed:
            changed = changed.replace(old, new, 1)
            used += 1
            if used == count:
                return changed
    return None


def build_correction_pool(songs: list[CleanedSong], count: int) -> list[EvalCase]:
    scenario = "Correction-one" if count == 1 else "Correction-two"
    cases: list[EvalCase] = []
    for base in build_single_sentence_cases(songs):
        typo_input = apply_typos(base.input_text, count)
        if typo_input and typo_input != base.input_text:
            cases.append(EvalCase(scenario, typo_input, base.expected, base.source))
    return cases


def sample_cases(cases: list[EvalCase], size: int, seed: int) -> list[EvalCase]:
    if len(cases) <= size:
        return list(cases)
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(cases)), size))
    return [cases[index] for index in indexes]


def build_recall_cases(songs: list[CleanedSong], config: AppConfig, seed: int) -> dict[str, list[EvalCase]]:
    token_count = make_token_counter(config)
    return {
        "Multi-sentences": build_multi_sentence_cases(songs, token_count),
        "Single sentence": build_single_sentence_cases(songs),
        "Complex context": build_complex_context_cases(songs, seed),
    }


def build_situation_cases(songs: list[CleanedSong], sample_size: int, seed: int) -> dict[str, list[EvalCase]]:
    pools = {
        "Half-sentences": build_half_sentence_pool(songs),
        "Symbols Outputs": build_symbol_pool(songs),
        "Correction-one": build_correction_pool(songs, 1),
        "Correction-two": build_correction_pool(songs, 2),
    }
    return {
        name: sample_cases(cases, sample_size, seed + offset)
        for offset, (name, cases) in enumerate(pools.items())
    }


def evaluate_cases(generator: LyricGenerator, cases: list[EvalCase]) -> EvalResult:
    correct = 0
    wrong = 0
    for case in cases:
        prediction = generator.predict(case.input_text)
        correct += int(exact_match(case.expected, prediction.text, prediction.accepted))
        wrong += int(is_wrong_output(case.expected, prediction.text, prediction.accepted))
    return EvalResult(correct=correct, wrong=wrong, total=len(cases))


def evaluate_modes(
    config: AppConfig,
    modes: list[str],
    strictnesses: list[str],
    cases_by_scenario: dict[str, list[EvalCase]],
) -> dict[str, dict[str, EvalResult]]:
    results: dict[str, dict[str, EvalResult]] = {scenario: {} for scenario in cases_by_scenario}
    generators: dict[str, LyricGenerator] = {}
    for mode in modes:
        normalized = mode.replace("_", "-").lower()
        for strictness in strictnesses:
            column = result_column(mode, strictness)
            for scenario, cases in cases_by_scenario.items():
                if normalized == "transformer":
                    results[scenario][column] = EvalResult(0, 0, len(cases), skipped=True)
                    continue
                if normalized not in generators:
                    generators[normalized] = LyricGenerator(config, mode=normalized)
                results[scenario][column] = evaluate_cases_with_strictness(generators[normalized], cases, strictness)
    return results


def evaluate_cases_with_strictness(generator: LyricGenerator, cases: list[EvalCase], strictness: str) -> EvalResult:
    correct = 0
    wrong = 0
    for case in cases:
        prediction = generator.predict(case.input_text, strictness=strictness)
        correct += int(exact_match(case.expected, prediction.text, prediction.accepted))
        wrong += int(is_wrong_output(case.expected, prediction.text, prediction.accepted))
    return EvalResult(correct=correct, wrong=wrong, total=len(cases))


def write_sample_manifest(path: Path, recall_cases: dict[str, list[EvalCase]], situation_cases: dict[str, list[EvalCase]]) -> None:
    payload = {
        "recall": {name: [case.__dict__ for case in cases] for name, cases in recall_cases.items()},
        "situations": {name: [case.__dict__ for case in cases] for name, cases in situation_cases.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_target_manifest(config: AppConfig) -> None:
    manifest_path = config.paths.processed_dir / "source_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            "Missing data/processed/source_manifest.json. Re-run prepare with both target --source-dir values."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sources = {Path(item.get("source_dir", "")).as_posix().replace("\\", "/") for item in data.get("sources", [])}
    root = config.paths.raw_dir.parent.parent
    expected_sources = {(root / source).as_posix() for source in TARGET_SOURCE_DIRS}
    if actual_sources != expected_sources:
        expected = ", ".join(sorted(expected_sources))
        actual = ", ".join(sorted(actual_sources))
        raise SystemExit(
            "Processed dataset does not match testresult.md. "
            f"Expected sources: {expected}. Actual sources: {actual}. "
            "Re-run prepare/train/calibrate/ngram before evaluation."
        )
    stats = data.get("stats", {})
    if int(stats.get("files", 0)) != 171:
        raise SystemExit(
            f"Expected 171 source files for testresult.md, found {stats.get('files')}. "
            "Re-run prepare/train/calibrate/ngram before evaluation."
        )


def result_column(mode: str, strictness: str) -> str:
    return f"{mode}:{strictness}"


def markdown_table(
    rows: list[str],
    columns: list[str],
    results: dict[str, dict[str, EvalResult]],
    metric: str,
) -> str:
    header = f"|                 | {' | '.join(columns)} |"
    separator = f"|{'-' * 17}|{'|'.join('-' * (len(column) + 2) for column in columns)}|"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            result = results.get(row, {}).get(column, EvalResult(0, 0, 0))
            values.append(result.display_wrong() if metric == "wrong" else result.display_accuracy())
        lines.append(f"| {row:<15} | {' | '.join(values)} |")
    return "\n".join(lines)


def render_report(
    recall_results: dict[str, dict[str, EvalResult]] | None,
    situation_results: dict[str, dict[str, EvalResult]] | None,
    columns: list[str],
) -> str:
    sections = ["# Generated Results", ""]
    if recall_results is not None:
        sections.extend(
            [
                "## Recall",
                "",
                "### Accuracy",
                "",
                markdown_table(["Multi-sentences", "Single sentence", "Complex context"], columns, recall_results, "accuracy"),
                "",
                "### Wrong Outputs",
                "",
                markdown_table(["Multi-sentences", "Single sentence", "Complex context"], columns, recall_results, "wrong"),
                "",
            ]
        )
    if situation_results is not None:
        sections.extend(
            [
                "## ACC for Some Situations",
                "",
                "### Accuracy",
                "",
                markdown_table(
                    ["Half-sentences", "Symbols Outputs", "Correction-one", "Correction-two"],
                    columns,
                    situation_results,
                    "accuracy",
                ),
                "",
                "### Wrong Outputs",
                "",
                markdown_table(
                    ["Half-sentences", "Symbols Outputs", "Correction-one", "Correction-two"],
                    columns,
                    situation_results,
                    "wrong",
                ),
                "",
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def parse_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"transformer", "model-only", "auto"}
    invalid = [mode for mode in modes if mode.replace("_", "-").lower() not in allowed]
    if invalid:
        raise SystemExit(f"Unsupported modes: {', '.join(invalid)}")
    return modes


def parse_strictnesses(raw: str) -> list[str]:
    strictnesses = [part.strip().replace("_", "-").lower() for part in raw.split(",") if part.strip()]
    allowed = {"strict", "balanced", "tolerant"}
    invalid = [strictness for strictness in strictnesses if strictness not in allowed]
    if invalid:
        raise SystemExit(f"Unsupported strictness values: {', '.join(invalid)}")
    return strictnesses


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LyricPredict for testresult.md tables.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", choices=("recall", "situations", "all"), default="all")
    parser.add_argument("--modes", default="transformer,model-only,auto")
    parser.add_argument("--strictnesses", default="strict,balanced,tolerant")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output", default="testresult.generated.md")
    parser.add_argument("--sample-output", default="data/processed/testresult_samples.json")
    parser.add_argument("--skip-manifest-check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.skip_manifest_check:
        validate_target_manifest(config)
    songs = _load_processed_songs(config.paths.processed_dir)
    if not songs:
        raise SystemExit("No processed songs found. Run prepare first.")

    modes = parse_modes(args.modes)
    strictnesses = parse_strictnesses(args.strictnesses)
    columns = [result_column(mode, strictness) for mode in modes for strictness in strictnesses]
    recall_cases = build_recall_cases(songs, config, args.seed) if args.section in {"recall", "all"} else {}
    situation_cases = (
        build_situation_cases(songs, args.sample_size, args.seed) if args.section in {"situations", "all"} else {}
    )
    write_sample_manifest(Path(args.sample_output), recall_cases, situation_cases)

    recall_results = evaluate_modes(config, modes, strictnesses, recall_cases) if recall_cases else None
    situation_results = evaluate_modes(config, modes, strictnesses, situation_cases) if situation_cases else None
    report = render_report(recall_results, situation_results, columns)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
