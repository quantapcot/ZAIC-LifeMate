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
    return {"id": str(i), "audio": "fixture.wav", "audio_text": text, "context": {}, "history": [], "label": {"tool_calls": [{"name": "demo.run", "arguments": {}}] * count, "tool_results": [{}] * count, "final_state": {}}}


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



class HardeningTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        import scripts.prepare_toolloop_data as module
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.patcher = patch.object(module, "ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.train = self.root / "inputs" / "train.jsonl"
        self.train.parent.mkdir()
        self.cat = self.train.parent / "catalog.json"
        self.records = [row(i, f"mẫu {i}", i % 3) for i in range(30)]
        self.train.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.records), encoding="utf-8")
        self.cat.write_text(json.dumps(catalog()), encoding="utf-8")
        self.registry = ToolRegistry.from_catalog(self.cat)
        self.out = self.root / "outputs" / "toolloop"

    def _prepare(self):
        return prepare(self.records, self.registry, self.train, self.cat, self.out)

    def _access(self):
        from src.toolcall.data.train_only import TrainOnlyDataset
        return TrainOnlyDataset.from_prepared_split(self.train, self.cat, self.out / "splits")

    def test_train_only_and_held_out(self):
        self._prepare()
        ids = {p: json.loads((self.out / "splits" / (p+"_ids.json")).read_text()) for p in ("train", "val", "test")}
        data = self._access()
        self.assertEqual({r['id'] for r in data.iter_records()}, set(ids['train']))
        sample = data.get(ids['train'][0]); sample['context']['modified'] = True
        self.assertNotIn('modified', data.get(ids['train'][0])['context'])
        for rid in (ids['val'][0], ids['test'][0], 'not-found'):
            with self.assertRaises(KeyError): data.get(rid)
        from src.toolcall.data.train_only import TrainOnlyDataset
        with self.assertRaises(ValueError): TrainOnlyDataset(self.records)

    def test_hash_mismatch_dataset_catalog(self):
        self._prepare()
        for path in (self.train, self.cat):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.assertRaises(ValueError): self._access()
            path.write_bytes(original)

    def test_bad_manifest_overlap_incomplete_and_unknown(self):
        self._prepare()
        directory = self.out / 'splits'
        manifest = directory / 'split_manifest.json'
        original = manifest.read_text()
        manifest.write_text('{')
        with self.assertRaises(ValueError): self._access()
        manifest.write_text(original)
        train, val = directory/'train_ids.json', directory/'val_ids.json'
        train_original, val_original = train.read_text(), val.read_text()
        ids, other = json.loads(train_original), json.loads(val_original)
        for bad in (ids + [other[0]], ids[:-1], ids[:-1] + ['unknown'], ids + [ids[0]], ids[:-1] + [None]):
            train.write_text(json.dumps(bad))
            # Update count to ensure coverage/overlap validation, not only count checks.
            m = json.loads(original); m['counts']['train'] = len(bad); manifest.write_text(json.dumps(m))
            with self.assertRaises(ValueError): self._access()
        train.write_text(train_original); val.write_text(val_original); manifest.write_text(original)
        # Swapping IDs preserves coverage and counts but violates deterministic assignments.
        ids[0], other[0] = other[0], ids[0]
        train.write_text(json.dumps(ids)); val.write_text(json.dumps(other))
        with self.assertRaises(ValueError): self._access()

    def test_dataset_validation(self):
        import copy
        malformed = []
        for field, value in [('context', None), ('history', {}), ('audio', None)]:
            item = copy.deepcopy(self.records[0]); item[field] = value; malformed.append(item)
        for field, value in [('tool_calls', [{'name': 'demo.run', 'arguments': []}]), ('tool_results', {}), ('final_state', [])]:
            item = copy.deepcopy(self.records[0]); item['label'][field] = value; malformed.append(item)
        item = copy.deepcopy(self.records[0]); item['label']['tool_calls'] = [{'name':'unknown', 'arguments':{}}]; malformed.append(item)
        item = copy.deepcopy(self.records[0]); item['history'] = ['invalid']; malformed.append(item)
        for item in malformed:
            self.train.write_text(json.dumps(item), encoding='utf-8')
            with self.assertRaises(ValueError): load_records(self.train, self.registry)
        self.train.write_text(json.dumps(self.records[0])+'\n'+json.dumps(self.records[0]), encoding='utf-8')
        with self.assertRaises(ValueError): load_records(self.train, self.registry)

    def test_leakage_deterministic_no_reassignment(self):
        from src.toolcall.data.leakage import leakage_audit
        data = [row('a', 'Xin chào!'), row('b', 'xin chào'), row('c', 'set appointment with the doctor at 10 in the morning'), row('d', 'set appointment with the doctor at 11 in the morning')]
        splits = {'train':['a','c'], 'val':['b','d'], 'test':[]}
        before = json.dumps(splits, sort_keys=True)
        result = leakage_audit(data, splits)
        self.assertEqual(result['cross_split_exact_normalized_pairs'], 1)
        self.assertGreaterEqual(result['cross_split_number_template_pairs'], 1)
        self.assertGreaterEqual(result['cross_split_lexically_similar_pairs'], 1)
        self.assertEqual(result, leakage_audit(list(reversed(data)), splits))
        self.assertEqual(before, json.dumps(splits, sort_keys=True))

    def test_output_safety_and_failed_publish_cleanup(self):
        from unittest.mock import patch
        self.out.parent.mkdir(parents=True)
        unrelated = self.out.parent/'keep.txt'; unrelated.write_text('keep')
        before = (self.train.read_bytes(), self.cat.read_bytes())
        with patch('scripts.prepare_toolloop_data._publish_directory', side_effect=OSError('simulated')):
            with self.assertRaises(OSError): self._prepare()
        self.assertFalse(self.out.exists())
        self.assertEqual(unrelated.read_text(), 'keep')
        self.assertFalse(list(self.out.parent.glob('.toolloop-preparing-*')))
        self.assertEqual(before, (self.train.read_bytes(), self.cat.read_bytes()))
        self._prepare()
        with self.assertRaises(FileExistsError): self._prepare()
        self.assertEqual(unrelated.read_text(), 'keep')
        bad = self.out/'unpublished'; broken = self.records[:-1]
        with self.assertRaises(ValueError): prepare(broken, self.registry, self.train, self.cat, bad)
        self.assertFalse(bad.exists())

    def test_registry_validation_and_immutability(self):
        import copy
        data = catalog(); before = copy.deepcopy(data)
        reg = ToolRegistry(data)
        self.assertEqual(data, before)
        data['apps'][0]['tools'][0]['parameters']['required'] = ['x', 'x']
        with self.assertRaises(ValueError): ToolRegistry(data)
        data = catalog(); data['apps'][0]['id'] = ''
        with self.assertRaises(ValueError): ToolRegistry(data)
        data = catalog(); data['apps'][0]['tools'][0]['parameters']['properties']['x']['type'] = 2
        with self.assertRaises(ValueError): ToolRegistry(data)
        self.assertEqual(reg.get_schema('demo.run')['properties']['y']['x-custom'], {'a':1})


if __name__ == "__main__":
    unittest.main()
