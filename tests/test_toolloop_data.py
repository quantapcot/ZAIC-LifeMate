import json
import tempfile
import unittest
from pathlib import Path
from src.toolcall.data import ToolRegistry
from src.toolcall.data.contracts import GroundTruth, Metadata, ToolCall, ForwardCalls, to_dict
from scripts.prepare_toolloop_data import load_records, make_splits, prepare


def catalog():
    return {"schema_version": "fixture", "apps": [{"id": "demo", "group": "example", "tools": [{"name": "demo.run", "parameters": {"type": "object", "properties": {"x": {"type": "string", "enum": ["a"]}, "y": {"type": "array", "items": {"type": "number"}, "x-custom": {"a": 1}}}, "required": ["x"], "x-one-of": [["x", "y"]]}}]}]}


def row(i, text, count=1):
    return {"id": str(i), "audio_text": text, "label": {"tool_calls": [{"name": "demo.run", "arguments": {}}] * count}}


class RegistryTests(unittest.TestCase):
    def test_catalog_lookup_constraints_and_arguments(self):
        data = catalog()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "catalog.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            r = ToolRegistry.from_catalog(p)
        self.assertEqual(r.list_tools(app="demo", category="example"), ["demo.run"])
        self.assertEqual(r.arguments("demo.run"), {"required": ["x"], "optional": ["y"]})
        self.assertEqual(r.get_schema("demo.run"), data["apps"][0]["tools"][0]["parameters"])
        copy = r.get_schema("demo.run")
        copy["required"].clear()
        self.assertEqual(r.arguments("demo.run")["required"], ["x"])
        with self.assertRaises(KeyError): r.get_tool("unknown")

    def test_duplicates_and_missing_schema(self):
        data = catalog()
        data["apps"][0]["tools"].append(data["apps"][0]["tools"][0])
        with self.assertRaises(ValueError): ToolRegistry(data)
        data = catalog()
        del data["apps"][0]["tools"][0]["parameters"]
        with self.assertRaises(ValueError): ToolRegistry(data)

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError): ToolRegistry.from_catalog(p)


class DatasetTests(unittest.TestCase):
    def test_split_stable_disjoint_and_query_grouped(self):
        records = [row(i, f"query {i}", i % 3) for i in range(30)] + [row("a", "Xin chào!"), row("b", "xin  chào")]
        splits = make_splits(records)
        self.assertEqual(splits, make_splits(list(reversed(records))))
        sets = [set(ids) for ids in splits.values()]
        self.assertEqual(len(set.union(*sets)), len(records))
        self.assertTrue(all(not a & b for i,a in enumerate(sets) for b in sets[i+1:]))
        self.assertTrue(any({"a", "b"} <= s for s in sets))

    def test_missing_or_invalid_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"train.jsonl"
            for text in ("{", '{}', json.dumps(row(1, 'x'))+'\n'+json.dumps(row(1, 'y'))):
                p.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError): load_records(p)

    def test_no_writing_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"train.jsonl"
            p.write_text("original", encoding="utf-8")
            with self.assertRaises(ValueError): prepare([row(1, "x")], ToolRegistry(catalog()), p, p, output=p.parent)
            self.assertEqual(p.read_text(encoding="utf-8"), "original")

    def test_contract_json(self):
        meta = Metadata("v1", "synthetic")
        ground = GroundTruth("single_tool", ["demo.run"], ["demo.run"], meta)
        calls = ForwardCalls("query", [ToolCall("demo.run", {"x": "a"})], meta)
        self.assertEqual(json.loads(json.dumps(to_dict(ground)))["metadata"]["validation_status"], "pending")
        self.assertEqual(json.loads(json.dumps(to_dict(calls)))["tool_calls"][0]["arguments"], {"x":"a"})


if __name__ == '__main__': unittest.main()
