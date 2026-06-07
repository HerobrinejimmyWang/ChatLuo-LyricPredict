from __future__ import annotations

import argparse
import re
import sys
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
    source: str


def normalize_for_compare(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.replace(",", "，").replace(".", "。")


def load_level_a_cases(path: Path) -> list[Case]:
    content = path.read_text(encoding="utf-8")
    level_a = content.split("### Level A", 1)[1].split("### Level B", 1)[0]
    cases: list[Case] = []
    for raw_line in level_a.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line or "NO" in line:
            continue
        columns = [part.strip() for part in line.strip("|").split("|")]
        if len(columns) < 6:
            continue
        no, subgroup = columns[0], columns[1]
        input_text, expected, source = columns[3], columns[4], columns[5]
        if subgroup.startswith("A-"):
            cases.append(Case(no=no, subgroup=subgroup, input_text=input_text, expected=expected, source=source))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LyricPredict on testcase.md Level A cases.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--testcase", default="testcase.md")
    parser.add_argument("--mode", choices=("auto", "model-only", "retrieval"), default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    generator = LyricGenerator(
        config,
        mode=args.mode,
        model_fallback_after_retrieval=False if args.mode == "retrieval" else None,
    )
    cases = load_level_a_cases(Path(args.testcase))
    if not cases:
        raise SystemExit("No Level A cases found.")

    passed = 0
    for case in cases:
        prediction = generator.predict(case.input_text)
        expected_key = normalize_for_compare(case.expected)
        actual_key = normalize_for_compare(prediction.text)
        ok = prediction.accepted and actual_key == expected_key
        passed += int(ok)
        if not args.quiet:
            status = "PASS" if ok else "FAIL"
            print(
                f"{status} {case.subgroup}-{case.no}: "
                f"expected={case.expected!r} actual={prediction.text!r} reason={prediction.reason}"
            )

    rate = passed / len(cases)
    print(f"Level A pass rate: {passed}/{len(cases)} = {rate:.1%}")
    return 0 if rate >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
