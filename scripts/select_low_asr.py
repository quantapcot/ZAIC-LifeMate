#!/usr/bin/env python3
"""Select or merge low-confidence transcripts through the ASR module."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.select_low_asr import main


if __name__ == "__main__":
    main()
