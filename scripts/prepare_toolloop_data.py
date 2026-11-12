"""Audit local train data and make grouped, deterministic, local-only ID splits."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import os
import shutil
import tempfile

from src.toolcall.data import ToolRegistry
from src.toolcall.data.dataset import load_records
from src.toolcall.data.split import query_key, make_splits
from src.toolcall.data.leakage import leakage_audit

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TRAIN = ROOT / "train"
DEFAULT_TRAIN = LOCAL_TRAIN if (LOCAL_TRAIN / "train.jsonl").is_file() else ROOT.parent.parent / "train"
OUTPUT = ROOT / "outputs" / "toolloop"



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


def _publish_directory(source, target):
    """Atomically publish without replacing an existing destination; fail closed."""
    if os.name == "nt":
        os.rename(source, target)  # Windows rename refuses an existing destination.
        return
    if os.name == "posix":
        import ctypes
        import errno
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError("atomic no-replace directory rename is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), str(target))
            raise OSError(error, os.strerror(error), str(target))
        return
    raise OSError("atomic no-replace directory rename is unavailable")


def prepare(records, registry, input_path, catalog_path, output=OUTPUT):
    output = Path(output).resolve()
    # Outputs must stay within this worktree, and may not overwrite existing files.
    allowed = ROOT / "outputs" / "toolloop"
    if allowed.resolve() != allowed or not output.is_relative_to(allowed):
        raise ValueError("output must be within outputs/toolloop")
    if any(output == Path(p).resolve() or Path(p).resolve().is_relative_to(output) or output.is_relative_to(Path(p).resolve().parent) for p in (input_path, catalog_path)):
        raise ValueError("output overlaps an input")
    input_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    catalog_hash = hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest()
    actual_registry = ToolRegistry.from_catalog(catalog_path)
    verified_records = load_records(input_path, actual_registry)
    if verified_records != records or actual_registry.list_tools() != registry.list_tools() or actual_registry.version != registry.version or any(actual_registry.get_tool(n) != registry.get_tool(n) for n in registry.list_tools()):
        raise ValueError("provided data does not match input files")
    splits = make_splits(records)
    stats = audit(records, registry)
    if stats["unknown_tool_names"]:
        raise ValueError("dataset contains tool names absent from catalog")
    manifest = {"seed": 42, "strategy": "NFC/casefold/punctuation-whitespace grouped; size and call-count stratified", "counts": {k:len(v) for k,v in splits.items()}, "dataset_sha256": input_hash, "catalog_sha256": catalog_hash, "catalog_version": registry.version}
    if output.exists():
        raise FileExistsError("preparation output already exists")
    # Stage in a unique sibling under the ignored output area, then publish in one rename.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".toolloop-preparing-", dir=output.parent))
    try:
        (temporary / "splits").mkdir()
        payloads = {f"splits/{key}_ids.json": ids for key, ids in splits.items()}
        payloads["splits/split_manifest.json"] = manifest
        stats["leakage"] = leakage_audit(records, splits)
        payloads["audit.json"] = stats
        for relative, value in payloads.items():
            path = temporary / relative
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError("output verification failed")
        if hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != input_hash or hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest() != catalog_hash:
            raise ValueError("inputs changed during preparation")
        if output.exists():
            raise FileExistsError("preparation output already exists")
        _publish_directory(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="new output directory under outputs/toolloop")
    args = parser.parse_args()
    dataset, catalog = args.train_dir / "train.jsonl", args.train_dir / "train_tools_catalog.json"
    registry = ToolRegistry.from_catalog(catalog)
    records = load_records(dataset, registry)
    stats = audit(records, registry)
    splits = make_splits(records)
    stats["leakage"] = leakage_audit(records, splits)
    print(json.dumps({"audit": stats, "planned_split_counts": {k:len(v) for k,v in splits.items()}, "seed": 42}, indent=2, ensure_ascii=True))
    if stats["unknown_tool_names"]:
        raise ValueError("dataset contains tool names absent from catalog")
    if args.prepare:
        prepare(records, registry, dataset, catalog, args.output_dir)
        print("Local outputs created in the selected ignored output directory.")


if __name__ == "__main__":
    main()
