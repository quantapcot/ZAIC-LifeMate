#!/usr/bin/env python3
"""Run one-pass LifeMate planning inference."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.plan_infer import main


if __name__ == "__main__":
    main()
