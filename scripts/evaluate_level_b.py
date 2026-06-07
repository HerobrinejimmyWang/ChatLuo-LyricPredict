from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lyricpredict.config import load_config
from lyricpredict.generation import LyricGenerator


@dataclass(frozen=True)
class Case:
    no: str
    subgroup: str
    input_text: str
    expected: str


def normalize_for_compare(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.replace(",", "，").replace(".", "。")


def output_matches(expected: str, actual: str) -> bool:
    expected_key = normalize_for_compare(expected)
    actual_key = normalize_for_compare(actual)
    if actual_key == expected_key:
        return True
    if expected_key[:1] in {"，", "。"} and actual_key == expected_key[1:]:
        return True
    return False


def load_level_b_cases(path: Path) -> list[Case]:
    content = path.read_text(encoding="utf-8")
    level_b = content.split("### Level B", 1)[1].split("### Level C", 1)[0]
    cases: list[Case] = []
    for raw_line in level_b.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line or "NO" in line:
            continue
        columns = [part.strip() for part in line.strip("|").split("|")]
        if len(columns) < 4:
            continue
        no, subgroup = columns[0], columns[1]
        if not subgroup.startswith("B-"):
            continue
        cases.append(Case(no=no, subgroup=subgroup, input_text=columns[2], expected=columns[3]))
    return cases


def case_passed(expected: str, actual: str, accepted: bool) -> bool:
    if expected == "无输出":
        return not accepted and actual == ""
    return accepted and output_matches(expected, actual)


def display_actual(actual: str, accepted: bool) -> str:
    return actual if accepted else "无输出"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LyricPredict on testcase.md Level B cases.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--testcase", default="testcase.md")
    parser.add_argument("--mode", choices=("auto", "model-only", "retrieval"), default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-errors-per-subgroup", type=int, default=1)
    args = parser.parse_args()

    generator = LyricGenerator(
        load_config(args.config),
        mode=args.mode,
        model_fallback_after_retrieval=False if args.mode == "retrieval" else None,
    )
    cases = load_level_b_cases(Path(args.testcase))
    if not cases:
        raise SystemExit("No Level B cases found.")

    errors: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for case in cases:
        prediction = generator.predict(case.input_text)
        ok = case_passed(case.expected, prediction.text, prediction.accepted)
        totals[case.subgroup] += 1
        errors[case.subgroup] += int(not ok)
        if not args.quiet:
            status = "PASS" if ok else "FAIL"
            print(
                f"{status} {case.subgroup}-{case.no}: "
                f"expected={case.expected!r} actual={display_actual(prediction.text, prediction.accepted)!r} "
                f"reason={prediction.reason}"
            )

    all_ok = True
    for subgroup in sorted(totals):
        err = errors[subgroup]
        total = totals[subgroup]
        all_ok = all_ok and err <= args.max_errors_per_subgroup
        print(f"{subgroup}: errors {err}/{total}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
