#!/usr/bin/env python3
"""Launch the local trading desk website."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def main() -> None:
    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8787,
        reload=True,
        reload_dirs=[str(ROOT / "web"), str(ROOT / "src")],
    )


if __name__ == "__main__":
    main()
