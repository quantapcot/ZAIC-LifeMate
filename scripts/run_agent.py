#!/usr/bin/env python3
"""Run iterative LifeMate agent inference."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.agent_infer import main


if __name__ == "__main__":
    main()
