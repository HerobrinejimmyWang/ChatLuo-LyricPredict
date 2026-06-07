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
    input_text: str
    lyric_context: str
    expected: str


def normalize_for_compare(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.replace(",", "，").replace(".", "。")


def normalize_continue_content(text: str) -> str:
    return re.sub(r"[\s,，.。]+", "", text)


def continuation_passed(actual: str, expected: str) -> bool:
    return normalize_continue_content(actual) == normalize_continue_content(expected)


def display_actual(actual: str, accepted_any: bool) -> str:
    return actual if accepted_any else "无输出"


def load_level_c_cases(path: Path) -> list[Case]:
    content = path.read_text(encoding="utf-8")
    level_c = content.split("### Level C", 1)[1].split("## 格式约束补充", 1)[0]
    cases: list[Case] = []
    for raw_line in level_c.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line or "NO" in line:
            continue
        columns = [part.strip() for part in line.strip("|").split("|")]
        if len(columns) < 4:
            continue
        cases.append(Case(no=columns[0], input_text=columns[1], lyric_context=columns[2], expected=columns[3]))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LyricPredict on testcase.md Level C cases.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--testcase", default="testcase.md")
    parser.add_argument("--mode", choices=("auto", "model-only", "retrieval"), default=None)
    parser.add_argument("--max-continues", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    generator = LyricGenerator(
        load_config(args.config),
        mode=args.mode,
        model_fallback_after_retrieval=False if args.mode == "retrieval" else None,
    )
    cases = load_level_c_cases(Path(args.testcase))
    if not cases:
        raise SystemExit("No Level C cases found.")

    passed = 0
    for case in cases:
        context = case.input_text
        outputs: list[str] = []
        reasons: list[str] = []
        ok = False
        for _ in range(max(1, args.max_continues)):
            prediction = generator.predict(context)
            reasons.append(prediction.reason)
            if not prediction.accepted:
                break
            outputs.append(prediction.text)
            actual = "".join(outputs)
            if normalize_for_compare(actual) == normalize_for_compare(case.expected) or continuation_passed(
                actual, case.expected
            ):
                ok = True
                break
            context = f"{context}{prediction.text}"
        passed += int(ok)
        if not args.quiet:
            status = "PASS" if ok else "FAIL"
            actual = "".join(outputs)
            print(
                f"{status} C-{case.no}: expected={case.expected!r} actual={display_actual(actual, bool(outputs))!r} "
                f"reasons={','.join(reasons)}"
            )

    print(f"Level C pass rate: {passed}/{len(cases)} = {passed / len(cases):.1%}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
