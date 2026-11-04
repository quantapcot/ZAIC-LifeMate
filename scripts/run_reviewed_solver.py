#!/usr/bin/env python3
"""Execute manually reviewed semantic plans."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.reviewed_solver import main


if __name__ == "__main__":
    main()
