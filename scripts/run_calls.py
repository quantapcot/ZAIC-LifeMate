#!/usr/bin/env python3
"""Execute reviewed tool calls through the agent module."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.run_calls import main


if __name__ == "__main__":
    main()
