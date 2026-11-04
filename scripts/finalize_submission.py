#!/usr/bin/env python3
"""Validate predictions and write the final submission."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.finalize_submission import main


if __name__ == "__main__":
    main()
