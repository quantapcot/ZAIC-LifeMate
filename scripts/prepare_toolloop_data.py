"""Audit local train data and make grouped, deterministic, local-only ID splits."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from src.toolcall.data import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TRAIN = ROOT / "train"
DEFAULT_TRAIN = LOCAL_TRAIN if (LOCAL_TRAIN / "train.jsonl").is_file() else ROOT.parent.parent / "train"
OUTPUT = ROOT / "outputs" / "toolloop"


def query_key(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text).split())


def load_records(path):
    records = []
    ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("audio_text"), str) or not isinstance(row.get("label"), dict) or not isinstance(row["label"].get("tool_calls"), list):
                raise ValueError(f"missing or malformed required fields at line {number}")
            if row["id"] in ids:
                raise ValueError(f"duplicate record ID at line {number}")
            ids.add(row["id"])
            records.append(row)
    if not records:
        raise ValueError("empty dataset")
    return records


def make_splits(records, seed=42):
    groups = defaultdict(list)
    for row in records:
        q = query_key(row["audio_text"])
        groups[q if q else f"\0{row['id']}"] .append(row)
    # Large groups first. Seeded tie-breaking, independent of input order.
    rng = random.Random(seed)
    group_list = [sorted(g, key=lambda r: r["id"]) for _, g in sorted(groups.items())]
    rng.shuffle(group_list)
    group_list.sort(key=len, reverse=True)
    fractions = {"train": .8, "val": .1, "test": .1}
    outputs = {key: [] for key in fractions}
    category_total = Counter(min(len(r["label"]["tool_calls"]), 2) for r in records)
    category_counts = {key: Counter() for key in fractions}
    for group in group_list:
        categories = Counter(min(len(r["label"]["tool_calls"]), 2) for r in group)
        def cost(key):
            # Normalized change in squared size and scenario deviations.
            def delta(current, increment, target):
                return ((current + increment - target) ** 2 - (current - target) ** 2) / max(target, 1)
            return delta(len(outputs[key]), len(group), fractions[key] * len(records)) + sum(delta(category_counts[key][c], n, fractions[key] * category_total[c]) for c, n in categories.items())
        chosen = min(fractions, key=cost)
        outputs[chosen].extend(r["id"] for r in group)
        category_counts[chosen].update(categories)
    return {key: sorted(value) for key, value in outputs.items()}


def audit(records, registry):
    fields = sorted(set().union(*(r.keys() for r in records)))
    types = {k: dict(Counter(type(r[k]).__name__ if k in r else "missing" for r in records)) for k in fields}
    empty = {k: sum(k not in r or r[k] is None for r in records) for k in fields}
    contexts = defaultdict(Counter)
    history_roles = Counter()
    history_lengths = Counter()
    calls = Counter()
    names = Counter()
    result_lengths = Counter()
    final_types = Counter()
    for row in records:
        for key, value in (row.get("context") or {}).items():
            contexts[key][type(value).__name__] += 1
        history = row.get("history") or []
        history_lengths[str(len(history))] += 1
        history_roles.update(str(h.get("role", "missing")) for h in history if isinstance(h, dict))
        label = row["label"]
        n = len(label["tool_calls"])
        calls[str(n)] += 1
        for call in label["tool_calls"]:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                raise ValueError("malformed tool call")
            names[call["name"]] += 1
        result_lengths[str(len(label.get("tool_results", [])))] += 1
        final_types[type(label.get("final_state")).__name__] += 1
    exact = Counter(r["audio_text"] for r in records)
    normalized = Counter(query_key(r["audio_text"]) for r in records)
    known = set(registry.list_tools())
    all_tools = [registry.get_tool(n) for n in registry.list_tools()]
    def walk(value, counter):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ("oneOf", "anyOf", "allOf", "enum", "items") or k.startswith("x-"):
                    counter[k] += 1
                walk(v, counter)
        elif isinstance(value, list):
            for item in value: walk(item, counter)
    constraints = Counter()
    for t in all_tools: walk(t["parameters"], constraints)
    return {"samples": len(records), "fields": types, "missing_or_null": empty,
            "duplicate_ids": 0, "duplicate_exact_queries_extra": sum(n-1 for n in exact.values()),
            "duplicate_normalized_queries_extra": sum(n-1 for n in normalized.values()),
            "call_counts": dict(calls), "tool_names": dict(sorted(names.items())),
            "unknown_tool_names": sorted(set(names) - known), "missing_info_calls": names.get("sys.ask_missing_info", 0),
            "context_types": {k: dict(v) for k,v in contexts.items()}, "history_lengths": dict(history_lengths),
            "history_roles": dict(history_roles), "result_counts": dict(result_lengths), "final_state_types": dict(final_types),
            "catalog_tools": len(all_tools), "catalog_version": registry.version,
            "catalog_app_tool_counts": dict(Counter(registry.get_tool(n).get("app", n.split(".")[0]) for n in registry.list_tools())),
            "catalog_kinds": dict(Counter(t.get("kind", "unspecified") for t in all_tools)),
            "schema_constraints": dict(constraints),
            "required_argument_count": sum(len(t["parameters"]["required"]) for t in all_tools),
            "optional_argument_count": sum(len(t["parameters"]["properties"]) - len(t["parameters"]["required"]) for t in all_tools)}


def prepare(records, registry, input_path, catalog_path, output=OUTPUT):
    output = Path(output).resolve()
    # Outputs must stay within this worktree, and may not overwrite existing files.
    if not output.is_relative_to((ROOT / "outputs" / "toolloop").resolve()):
        raise ValueError("output must be within outputs/toolloop")
    if any(output == Path(p).resolve() or Path(p).resolve().is_relative_to(output) or output.is_relative_to(Path(p).resolve().parent) for p in (input_path, catalog_path)):
        raise ValueError("output overlaps an input")
    splits = make_splits(records)
    stats = audit(records, registry)
    if stats["unknown_tool_names"]:
        raise ValueError("dataset contains tool names absent from catalog")
    manifest = {"seed": 42, "strategy": "NFC/casefold/punctuation-whitespace grouped; size and call-count stratified", "counts": {k:len(v) for k,v in splits.items()}, "dataset_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(), "catalog_sha256": hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(), "catalog_version": registry.version}
    output.mkdir(parents=True, exist_ok=False)
    (output / "splits").mkdir()
    for key, ids in splits.items():
        (output / "splits" / f"{key}_ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "splits" / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    args = parser.parse_args()
    dataset, catalog = args.train_dir / "train.jsonl", args.train_dir / "train_tools_catalog.json"
    registry = ToolRegistry.from_catalog(catalog)
    records = load_records(dataset)
    stats = audit(records, registry)
    splits = make_splits(records)
    print(json.dumps({"audit": stats, "planned_split_counts": {k:len(v) for k,v in splits.items()}, "seed": 42}, indent=2, ensure_ascii=True))
    if stats["unknown_tool_names"]:
        raise ValueError("dataset contains tool names absent from catalog")
    if args.prepare:
        prepare(records, registry, dataset, catalog)
        print("Local outputs created in outputs/toolloop (ignored by Git).")


if __name__ == "__main__":
    main()
