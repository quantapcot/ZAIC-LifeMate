"""Fail-closed access to the train portion of a prepared split.

This accidental-leakage boundary is not a sandbox against direct file access.
"""
import copy
import hashlib
import json
from pathlib import Path

from .dataset import load_records
from .schema_registry import ToolRegistry
from .split import make_splits


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainOnlyDataset:
    __TOKEN = object()

    def __init__(self, records, token=None):
        if token is not self.__TOKEN:
            raise ValueError("use from_prepared_split")
        self.__records = {r["id"]: copy.deepcopy(r) for r in records}

    @classmethod
    def from_prepared_split(cls, train_path, catalog_path, splits_dir):
        train_path, catalog_path, splits_dir = Path(train_path), Path(catalog_path), Path(splits_dir)
        try:
            manifest = json.loads((splits_dir / "split_manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(manifest.get("counts"), dict):
                raise ValueError("malformed split manifest")
            hashes = (_sha(train_path), _sha(catalog_path))
            if (manifest.get("dataset_sha256"), manifest.get("catalog_sha256")) != hashes:
                raise ValueError("dataset or catalog hash mismatch")
            registry = ToolRegistry.from_catalog(catalog_path)
            if manifest.get("catalog_version") != registry.version:
                raise ValueError("catalog version mismatch")
            sets = {}
            for part in ("train", "val", "test"):
                ids = json.loads((splits_dir / f"{part}_ids.json").read_text(encoding="utf-8"))
                if not isinstance(ids, list) or any(not isinstance(i, str) or not i.strip() or i != i.strip() for i in ids) or len(set(ids)) != len(ids):
                    raise ValueError(f"malformed or duplicate {part} IDs")
                if type(manifest["counts"].get(part)) is not int or manifest["counts"][part] != len(ids):
                    raise ValueError(f"manifest count mismatch for {part}")
                sets[part] = set(ids)
            if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
                raise ValueError("overlapping split IDs")
            records = load_records(train_path, registry)
            all_ids = {r["id"] for r in records}
            if set.union(*sets.values()) != all_ids:
                raise ValueError("split coverage does not match dataset")
            if manifest.get("seed") != 42 or manifest.get("strategy") != "NFC/casefold/punctuation-whitespace grouped; size and call-count stratified":
                raise ValueError("unknown split strategy")
            expected = make_splits(records, seed=42)
            if any(sets[k] != set(expected[k]) for k in sets):
                raise ValueError("split assignments do not match manifest strategy")
            if (_sha(train_path), _sha(catalog_path)) != hashes:
                raise ValueError("inputs changed while reading")
            return cls([r for r in records if r["id"] in sets["train"]], cls.__TOKEN)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("missing or malformed preparation metadata or input") from exc

    def get(self, record_id):
        if record_id not in self.__records:
            raise KeyError("record is not available in the training split")
        return copy.deepcopy(self.__records[record_id])

    def iter_records(self):
        for record_id in sorted(self.__records):
            yield copy.deepcopy(self.__records[record_id])

    def __len__(self):
        return len(self.__records)
