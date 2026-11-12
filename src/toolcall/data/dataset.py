"""Read and validate the public structure of LifeMate training records."""
import json
from pathlib import Path


def load_records(path, registry=None):
    records, ids = [], set()
    known = set(registry.list_tools()) if registry is not None else None
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {number}") from exc
            def fail(field):
                raise ValueError(f"missing or malformed {field} at line {number}")
            if not isinstance(row, dict): fail("record")
            rid = row.get("id")
            if not isinstance(rid, str) or not rid.strip() or rid != rid.strip(): fail("id")
            if rid in ids: fail("duplicate id")
            ids.add(rid)
            if not isinstance(row.get("audio"), str) or not row["audio"].strip(): fail("audio")
            if not isinstance(row.get("audio_text"), str): fail("audio_text")
            if not isinstance(row.get("context"), dict): fail("context")
            context = row["context"]
            for key, expected in (("current_time", str), ("timezone", str), ("location", dict), ("user_profile", dict), ("installed_apps", dict)):
                if key in context and not isinstance(context[key], expected): fail(f"context.{key}")
            if not isinstance(row.get("history"), list): fail("history")
            def calls_valid(calls, field):
                if not isinstance(calls, list): fail(field)
                for call in calls:
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"].strip() or not isinstance(call.get("arguments"), dict): fail(field)
                    if known is not None and call["name"] not in known: fail(f"unknown tool in {field}")
            for entry in row["history"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("role"), str) or not entry["role"].strip(): fail("history entry")
                for key in ("audio", "audio_text", "text"):
                    if key in entry and not isinstance(entry[key], str): fail(f"history.{key}")
                if "tool_calls" in entry: calls_valid(entry["tool_calls"], "history.tool_calls")
                if "tool_results" in entry and (not isinstance(entry["tool_results"], list) or any(not isinstance(x, dict) for x in entry["tool_results"])): fail("history.tool_results")
            label = row.get("label")
            if not isinstance(label, dict): fail("label")
            calls_valid(label.get("tool_calls"), "label.tool_calls")
            if not isinstance(label.get("tool_results"), list) or any(not isinstance(x, dict) for x in label["tool_results"]): fail("label.tool_results")
            if not isinstance(label.get("final_state"), dict): fail("label.final_state")
            records.append(row)
    if not records: raise ValueError("empty dataset")
    return records
