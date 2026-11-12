import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from src.toolcall.data import ToolRegistry
from src.toolcall.data.contracts import GroundTruth, Metadata
from src.toolcall.llm import OllamaClient, TransportError, InvalidModelOutput
from src.toolcall.synthesis import sampler, validators, pipeline
from scripts.synthesize_toolcalls import write_run
from tests.test_toolloop_data import catalog, row


def registry():
    c=catalog(); c['apps'][0]['tools'][0]['parameters'].pop('x-one-of'); c['apps'][0]['tools'].append({'name':'demo.other','kind':'read','parameters':{'type':'object','properties':{},'required':[]}})
    return ToolRegistry(c)


def record():
    r=row('train','example');r['context']['installed_apps']={'demo':{'tools':['demo.run','demo.other']}}
    return r


def ground(): return GroundTruth('multiple',['demo.run','demo.other'],['demo.run'],Metadata('fixture','train_split'))


class Response:
    def __init__(self, content): self.content=content
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,*args): return self.content


class ClientTests(unittest.TestCase):
    def test_json_and_metadata(self):
        client=OllamaClient(); client.opener=MagicMock()
        client.opener.open.return_value=Response(json.dumps({'done':True,'message':{'content':'{"ok":true}'},'eval_count':4}).encode())
        raw,usage=client.generate_json('system','user',{'type':'object'})
        self.assertEqual(raw,{'ok':True});self.assertEqual(usage['output_tokens'],4)
        payload=json.loads(client.opener.open.call_args.args[0].data)
        self.assertFalse(payload['stream']);self.assertEqual(payload['format'],{'type':'object'})

    def test_invalid_json_and_malformed(self):
        client=OllamaClient();client.opener=MagicMock()
        for body in (b'garbage', b'{}', b'{"done":true,"message":{"content":"no"}}', b'{"done":true,"message":{"content":"{\\"x\\":NaN}"}}'):
            client.opener.open.return_value=Response(body)
            with self.assertRaises(InvalidModelOutput): client.generate_json('s','u',{})

    def test_response_schema_is_checked(self):
        client=OllamaClient();client.opener=MagicMock()
        client.opener.open.return_value=Response(b'{"done":true,"message":{"content":"{\"ok\":1}"}}')
        with self.assertRaises(InvalidModelOutput):
            client.generate_json('s','u',{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']})

    def test_timeout_and_unavailable(self):
        client=OllamaClient();client.opener=MagicMock()
        for exc in (TimeoutError(),URLError('offline')):
            client.opener.open.side_effect=exc
            with self.assertRaises(TransportError):client.generate_json('s','u',{})

    def test_local_url_and_context_cap(self):
        for url in ('https://127.0.0.1:11434','http://example.com','http://localhost.evil','http://user@localhost:1','http://localhost/path'):
            with self.assertRaises(ValueError):OllamaClient(base_url=url)
        with self.assertRaises(ValueError):OllamaClient(context_length=16385)


class ValidatorTests(unittest.TestCase):
    def test_ground_validation(self):
        r=registry();g=ground();self.assertTrue(validators.validate_ground(g,r).valid)
        g.target_tools=['unknown'];self.assertFalse(validators.validate_ground(g,r).valid)
        g.target_tools=['demo.run'];g.candidate_tools=['demo.other'];self.assertFalse(validators.validate_ground(g,r).valid)
        g.scenario_type='parallel';self.assertFalse(validators.validate_ground(g,r).valid)

    def test_schema_failures(self):
        g=ground();r=registry()
        def check(args):return validators.validate_calls({'tool_calls':[{'name':'demo.run','arguments':args}]},g,r)
        self.assertTrue(check({'x':'a'}).valid)
        for args in ({},{'x':1},{'x':'a','extra':1},{'x':'b'},{'x':'a','y':['wrong']}):
            self.assertFalse(check(args).valid)
        self.assertFalse(validators.validate_calls({'tool_calls':[{'name':'unknown','arguments':{}}]},g,r).valid)

    def test_custom_conditional_and_unverified(self):
        c=catalog();schema=c['apps'][0]['tools'][0]['parameters'];schema['x-one-of']=[['x','y']];schema['properties']['x']['x-ref']='example'
        r=ToolRegistry(c);g=GroundTruth('simple',['demo.run'],['demo.run'],Metadata('v','train'))
        bad=validators.validate_calls({'tool_calls':[{'name':'demo.run','arguments':{'x':'a'}}]},g,r);self.assertFalse(bad.valid)
        good=validators.validate_calls({'tool_calls':[{'name':'demo.run','arguments':{'x':'a','y':[1]}}]},g,r)
        self.assertTrue(good.valid); self.assertIn({'name':'x-ref','status':'not_verified'},good.checks)


class FakeClient:
    model='fake'
    options={'seed':42}
    def __init__(self, fail=False):self.calls=[];self.fail=fail
    def generate_json(self, system_prompt,user_prompt,output_schema):
        self.calls.append(json.loads(user_prompt))
        usage={'latency_seconds':.01,'output_tokens':2}
        if self.fail: return {'wrong':True}, usage
        if len(self.calls)==1:return {'user_query':'Xin thực hiện việc này với giá trị a'},usage
        return {'tool_calls':[{'name':'demo.run','arguments':{'x':'a'}}]},usage


class PipelineTests(unittest.TestCase):
    def test_sampling_deterministic_and_unsupported(self):
        r=registry();self.assertEqual(sampler.sample(record(),r,'multiple',42),sampler.sample(record(),r,'multiple',42))
        with self.assertRaises(ValueError):sampler.sample(record(),r,'parallel',42)

    def test_order_and_no_answer_leak(self):
        c=FakeClient();result=pipeline.run_attempt(record(),registry(),c,'multiple',42)
        self.assertEqual(result['status'],'provisional');self.assertEqual(len(c.calls),2)
        self.assertIn('targets',c.calls[0]);self.assertNotIn('targets',c.calls[1]);self.assertIn('candidate_tools',c.calls[1])

    def test_failed_stage_stops_and_no_retry(self):
        c=FakeClient(fail=True);r=pipeline.run_attempt(record(),registry(),c,'simple',42)
        self.assertEqual(r['failure_stage'],2);self.assertEqual(len(c.calls),1)
        bad=record();bad['label']['tool_calls']=[];c=FakeClient();r=pipeline.run_attempt(bad,registry(),c,'simple',42)
        self.assertEqual(r['failure_stage'],1);self.assertEqual(len(c.calls),0)

    def test_training_copy_rejected(self):
        c=FakeClient();r=pipeline.run_attempt(record(),registry(),c,'simple',42,{'xin thuc hien viec nay voi gia tri a'})
        # Test exact normalized output using the same normalizer.
        from src.toolcall.data.split import query_key
        c=FakeClient();r=pipeline.run_attempt(record(),registry(),c,'simple',42,{query_key('Xin thực hiện việc này với giá trị a')})
        self.assertEqual(r['failure_stage'],2);self.assertEqual(len(c.calls),1)

    def test_training_view_required_and_no_held_out_lookup(self):
        from src.toolcall.data import TrainOnlyDataset
        class UnsafeDataset:
            def iter_records(self): return iter([record()])
        with self.assertRaises(TypeError):
            pipeline.generate(UnsafeDataset(),registry(),FakeClient(),1,42)
        view=MagicMock(spec=TrainOnlyDataset)
        view.iter_records.return_value=iter([record()])
        view.get.side_effect=AssertionError("unexpected lookup")
        result=pipeline.generate(view,registry(),FakeClient(),1,42)
        self.assertEqual(len(result),1)
        view.get.assert_not_called()

    def test_synthesis_does_not_import_or_execute_toolenv(self):
        import ast
        root=Path(__file__).resolve().parents[1]/'src'/'toolcall'/'synthesis'
        for file in root.glob('*.py'):
            tree=ast.parse(file.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(alias.name.startswith('zaic_tools') for alias in node.names))
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('zaic_tools'))
                if isinstance(node,ast.Name): self.assertNotEqual(node.id,'ToolEnv')

    def test_output_local_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); base=root/'synthesis'
            with patch('scripts.synthesize_toolcalls.SYNTHESIS_ROOT',base):
                path=base/'run';write_run(path,[],{})
                with self.assertRaises(FileExistsError):write_run(path,[],{})
                with self.assertRaises(ValueError):write_run(root/'outside',[],{})
                self.assertTrue((path/'COMPLETE').is_file())
