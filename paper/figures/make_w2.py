#!/usr/bin/env python3
"""Generate all CSV data for W2 (Sections 3 and 4: ring-*.csv and bounds-*.csv).

Run from repository root or manuscript/jmb:
    python3 -I ../paper/figures/make_w2.py
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

import make_bounds  # noqa: E402
import make_ring  # noqa: E402


def main() -> None:
    make_ring.main()
    make_bounds.main()


if __name__ == "__main__":
    main()
