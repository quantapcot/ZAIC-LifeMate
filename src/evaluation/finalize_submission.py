#!/usr/bin/env python3
"""Strip diagnostics, replay calls, and write a validated submission JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in (ROOT / "public_test/public_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predictions = {}
    for line in args.prediction.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            predictions[row["id"]] = row

    expected_ids = [item["id"] for item in items]
    missing = [sample_id for sample_id in expected_ids if sample_id not in predictions]
    extras = sorted(set(predictions) - set(expected_ids))
    if missing or extras:
        raise SystemExit(f"ID mismatch: missing={missing[:10]} extras={extras[:10]}")

    sys.path.insert(0, str(ROOT / "public_test"))
    from zaic_tools import ToolEnv

    final = []
    for sample_id in expected_ids:
        calls = predictions[sample_id].get("tool_calls") or []
        env = ToolEnv("public_test", sample_id, root=ROOT)
        for call in calls:
            validation = env.validate(call.get("name"), call.get("arguments") or {})
            if not validation["ok"]:
                raise SystemExit(f"Invalid call for {sample_id}: {call}: {validation['error']}")
            env.call(call["name"], call.get("arguments") or {})
        computed_hash = env.get_state_hash()
        stored_hash = predictions[sample_id].get("state_hash")
        if stored_hash and stored_hash != computed_hash:
            raise SystemExit(
                f"Hash mismatch for {sample_id}: stored={stored_hash} replay={computed_hash}"
            )
        final.append({"id": sample_id, "tool_calls": calls, "state_hash": computed_hash})

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
        encoding="utf-8",
    )
    print(f"Wrote {len(final)} validated rows to {args.output}")


if __name__ == "__main__":
    main()
