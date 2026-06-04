from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LyricPredict local Web app.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["LYRICPREDICT_CONFIG"] = args.config

    import uvicorn

    uvicorn.run("lyricpredict.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
