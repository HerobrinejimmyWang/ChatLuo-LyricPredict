from __future__ import annotations

import argparse

from .qt_app import run_desktop_app
from .windows_io import DesktopDependencyError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LyricPredict Windows desktop app.")
    parser.add_argument("--settings", default="configs/app.yaml")
    args = parser.parse_args()
    try:
        return int(run_desktop_app(args.settings))
    except DesktopDependencyError as exc:
        print(f"LyricPredict desktop app cannot start: {exc}")
        print("Install optional desktop dependencies with: pip install PySide6 pywin32 pyinstaller")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

