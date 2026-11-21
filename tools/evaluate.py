#!/usr/bin/env python3
"""PQ/SQ/RQ 本地评估入口（等价于 ``python -m cadvec.evaluate``）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.evaluate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
