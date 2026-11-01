#!/usr/bin/env python3
"""Export persistent agent sessions as a submission."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.export_result import main


if __name__ == "__main__":
    main()
