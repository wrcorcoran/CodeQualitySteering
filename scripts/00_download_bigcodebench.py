#!/usr/bin/env python3
"""Pre-cache BigCodeBench dataset. Run once on a login node before submitting jobs.

Usage:
    uv run python scripts/download_bigcodebench.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bigcodebench.data import get_bigcodebench

for subset in ("full", "hard"):
    print(f"Caching subset={subset}...")
    problems = get_bigcodebench(subset=subset)
    print(f"  {len(problems)} tasks cached.")