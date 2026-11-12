import copy
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from src.toolcall.data import TrainOnlyDataset
from src.toolcall.llm import OllamaClient, InvalidModelOutput, TransportError
from src.toolcall.synthesis import sampler, pipeline, validators
from src.toolcall.synthesis.schema_inspection import unsupported_constraints
from scripts import synthesize_toolcalls as cli
from tests.test_toolloop_synthesis import registry, record, ground, FakeClient, Response


class SchemaInspectionTests(unittest.TestCase):
    def test_recursive_constraints_and_immutability(self):
        cases = [
            ({'x-ref':'state'},'UNSUPPORTED_X_REF'),
            ({'properties':{'outer':{'properties':{'inner':{'x-ref':'state'}}}}},'UNSUPPORTED_X_REF'),
            ({'items':{'x-match':'state'}},'UNSUPPORTED_X_MATCH'),
            ({'properties':{'a':{'x-one-of':[['x']]}}},'UNSUPPORTED_CONDITIONAL_SCHEMA'),
            ({'oneOf':[{'x-ref':'state'}]},'UNSUPPORTED_X_REF'),
            ({'anyOf':[{'items':{'x-match':'state'}}]},'UNSUPPORTED_X_MATCH'),
            ({'allOf':[{'properties':{'x':{'x-one-of':[['x']]}}}]},'UNSUPPORTED_CONDITIONAL_SCHEMA')]
        for schema, expected in cases:
            original=copy.deepcopy(schema)
            result=unsupported_constraints(schema)
            self.assertIn(expected,[x['code'] for x in result])
            self.assertEqual(result,unsupported_constraints(schema))
            self.assertEqual(schema,original)
        self.assertEqual(unsupported_constraints({'type':'object','properties':{'x':{'type':'string','enum':['a']}}}),[])

    def test_sampler_rejects_nested_and_keeps_original(self):
        r=registry(); t=r._tools['demo.run']['parameters'];t['properties']['y']['items']['x-match']='state'
        original=r.get_schema('demo.run')
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_X_MATCH'):sampler.sample(record(),r,'simple',42)
        self.assertEqual(original,r.get_schema('demo.run'))


class StructuredErrorsTests(unittest.TestCase):
    def test_transport_and_model_error_codes(self):
        c=OllamaClient();c.opener=MagicMock()
        for exc,code in [(TimeoutError(),'TIMEOUT'),(URLError(socket.timeout()),'TIMEOUT'),(URLError('offline'),'TRANSPORT_FAILURE')]:
            c.opener.open.side_effect=exc
            with self.assertRaises(TransportError) as result:c.generate_json('s','u',{'type':'object'})
            self.assertEqual(result.exception.code,code)
        c.opener.open.side_effect=None
        for body,code in [(b'{}','MALFORMED_ENVELOPE'),(json.dumps({'done':True,'message':{'content':'not-json'}}).encode(),'INVALID_JSON'),(json.dumps({'done':True,'message':{'content':'{"x": 2}'}}).encode(),'OUTPUT_SCHEMA_MISMATCH')]:
            c.opener.open.return_value=Response(body)
            with self.assertRaises(InvalidModelOutput) as result:c.generate_json('s','u',{'type':'object','properties':{'x':{'type':'string'}},'required':['x']})
            self.assertEqual(result.exception.code,code)
            if code=='OUTPUT_SCHEMA_MISMATCH':self.assertEqual(result.exception.field,'$.x')

    def test_numeric_range_and_query_copy_codes(self):
        r=registry()
        r._tools['demo.run']['parameters']['properties']['y']['items']['minimum']=3
        result=validators.validate_calls({'tool_calls':[{'name':'demo.run','arguments':{'x':'a','y':[1]}}]},ground(),r)
        self.assertIn('INVALID_NUMERIC_RANGE',[item['error_code'] for item in result.issues])
        from src.toolcall.data.split import query_key
        reference=FakeClient().generate_json('', '{}', {})[0]['user_query']
        client=FakeClient()
        sample=pipeline.run_attempt(record(),registry(),client,'simple',42,{query_key(reference)})
        self.assertEqual(sample['feedback']['issues'][0]['error_code'],'QUERY_COPY')

    def test_error_details_retained_without_retry(self):
        class Broken(FakeClient):
            def generate_json(self,*args):
                self.calls.append('once')
                raise InvalidModelOutput('bad',code='INVALID_JSON',field='message.content',raw_output='local-only')
        c=Broken();result=pipeline.run_attempt(record(),registry(),c,'simple',42)
        problem=result['feedback']['issues'][0]
        self.assertEqual((result['failure_stage'],problem['error_code'],problem['field'],problem['source']), (2,'INVALID_JSON','message.content','model_output'))
        self.assertTrue(problem['retryable']);self.assertEqual(result['local_invalid_output_excerpt'],'local-only');self.assertEqual(len(c.calls),1)
        class Timeout(FakeClient):
            def generate_json(self,*args):raise TransportError('timeout',code='TIMEOUT')
        f=pipeline.run_attempt(record(),registry(),Timeout(),'simple',42)['feedback']['issues'][0]
        self.assertEqual((f['error_code'],f['source'],f['retryable']),('TIMEOUT','transport',False))

    def test_validator_actionable_issue_codes(self):
        r=registry();g=ground()
        for args,expected in [({},'MISSING_REQUIRED_ARGUMENT'),({'x':3},'INVALID_ARGUMENT_TYPE'),({'x':'bad'},'INVALID_ENUM'),({'x':'a','extra':1},'UNEXPECTED_ARGUMENT')]:
            result=validators.validate_calls({'tool_calls':[{'name':'demo.run','arguments':args}]},g,r)
            self.assertIn(expected,[x['error_code'] for x in result.issues])
            self.assertTrue(all('field' in x and 'message' in x and 'source' in x and 'retryable' in x for x in result.issues))
        result=validators.validate_calls({'tool_calls':[{'name':'unknown','arguments':{}}]},g,r)
        self.assertIn('INVALID_TOOL_NAME',[x['error_code'] for x in result.issues])
        result=validators.validate_calls({'tool_calls':[{'name':'demo.other','arguments':{}}]},g,r)
        self.assertIn('TARGET_MISMATCH',[x['error_code'] for x in result.issues])


class RunAccountingTests(unittest.TestCase):
    def _view(self, n):
        rows=[]
        for i in range(n):
            r=record();r['id']=str(i);rows.append(r)
        view=MagicMock(spec=TrainOnlyDataset);view.iter_records.side_effect=lambda:iter(rows)
        return view

    def test_sufficient_shortfall_zero_and_deterministic(self):
        # Avoid model calls while testing selection/accounting.
        def fake_attempt(record, registry, client, scenario, seed, queries):
            return {'status':'provisional','seed':seed,'scenario_type':scenario,'id':record['id']}
        with patch('src.toolcall.synthesis.pipeline.run_attempt',side_effect=fake_attempt):
            for size,expected in [(12,(12,10,0)),(3,(3,3,7)),(0,(0,0,10))]:
                records, report=pipeline.generate(self._view(size),registry(),FakeClient(),10,42,with_report=True)
                self.assertEqual((report['eligible'],report['attempted'],report['shortfall']),expected)
                again,_=pipeline.generate(self._view(size),registry(),FakeClient(),10,42,with_report=True)
                self.assertEqual(records,again)
                self.assertEqual(len({r['id'] for r in records}),len(records))

    def test_outcome_counts_and_transport_failure(self):
        ok={'status':'provisional','failure_reason':None}
        rejected={'status':'rejected','failure_reason':'invalid_calls'}
        transport={'status':'rejected','failure_reason':'TransportError'}
        self.assertEqual(cli.run_outcome([ok,rejected],{'shortfall':0}),'completed')
        self.assertEqual(cli.run_outcome([ok],{'shortfall':1}),'partial')
        self.assertEqual(cli.run_outcome([] ,{'shortfall':2}),'failed')
        self.assertEqual(cli.run_outcome([transport],{'shortfall':0}),'failed')

    def test_markers_and_failed_write(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'synthesis'
            with patch.object(cli,'SYNTHESIS_ROOT',root):
                for outcome,marker in [('completed','COMPLETE'),('partial','PARTIAL'),('failed','FAILED')]:
                    out=root/outcome;cli.write_run(out,[],{'outcome':outcome});self.assertTrue((out/marker).is_file())
                    if outcome!='completed': self.assertFalse((out/'COMPLETE').exists())
                with patch.object(cli.json,'dump',side_effect=OSError('disk error')):
                    with self.assertRaises(OSError):cli.write_run(root/'interrupted',[],{'outcome':'completed'})
                self.assertFalse((root/'interrupted'/'COMPLETE').exists())
                with patch.object(cli.os, 'replace', side_effect=OSError('rename failure')):
                    with self.assertRaises(OSError): cli.write_run(root/'marker-failure',[],{'outcome':'completed'})
                self.assertFalse((root/'marker-failure'/'COMPLETE').exists())
                with self.assertRaises(FileExistsError):cli.write_run(root/'completed',[],{})
                with self.assertRaises(ValueError):cli.write_run(root.parent/'escape',[],{})

    def test_generate_cli_refuses_existing_and_unsafe(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'synthesis'; existing=root/'run';existing.mkdir(parents=True)
            fake=MagicMock(); data=MagicMock()
            with patch.object(cli,'SYNTHESIS_ROOT',root),patch.object(cli,'OllamaClient',return_value=fake),patch.object(cli.ToolRegistry,'from_catalog',return_value=registry()),patch.object(cli.TrainOnlyDataset,'from_prepared_split',return_value=data),patch.object(cli,'generate') as generate,patch('sys.stdout',new_callable=io.StringIO),patch('sys.stderr',new_callable=io.StringIO):
                for target in (existing,root.parent/'escape'):
                    with patch('sys.argv',['tool','--generate','--output-dir',str(target)]):
                        with self.assertRaises(SystemExit) as result:cli.main()
                        self.assertEqual(result.exception.code,1)
                generate.assert_not_called()
                self.assertEqual(list(existing.iterdir()),[])

    def test_partial_cli_saves_record_and_signals_shortfall(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'synthesis';splits=Path(d)/'splits';splits.mkdir()
            (splits/'split_manifest.json').write_text(json.dumps({'dataset_sha256':'fixture'}))
            fake=MagicMock();fake.model='fake';fake.options={};fake.base_url='http://127.0.0.1:11434';fake.timeout=1;fake.model_digest.return_value=None
            records=[{'status':'provisional','scenario_type':'simple','failure_stage':None,'failure_reason':None,'stages':{}}]
            selection={'requested':2,'eligible':1,'attempted':1,'shortfall':1,'excluded_by_reason':{}}
            with patch.object(cli,'SYNTHESIS_ROOT',root),patch.object(cli,'OllamaClient',return_value=fake),patch.object(cli.ToolRegistry,'from_catalog',return_value=registry()),patch.object(cli.TrainOnlyDataset,'from_prepared_split'),patch.object(cli,'generate',return_value=(records,selection)),patch('sys.stdout',new_callable=io.StringIO),patch('sys.stderr',new_callable=io.StringIO):
                with patch('sys.argv',['tool','--generate','--limit','2','--splits-dir',str(splits),'--output-dir',str(root/'partial')]):
                    with self.assertRaises(SystemExit) as result:cli.main()
                self.assertEqual(result.exception.code,1)
            self.assertTrue((root/'partial'/'PARTIAL').is_file())
            self.assertFalse((root/'partial'/'COMPLETE').exists())
            saved=json.loads((root/'partial'/'summary.json').read_text(encoding='utf8'))
            self.assertEqual((saved['eligible'],saved['attempted'],saved['shortfall'],saved['outcome']),(1,1,1,'partial'))

    def test_read_only_modes_ignore_existing_output_and_dry_has_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'synthesis'; existing=root/'run';existing.mkdir(parents=True)
            fake=MagicMock();fake.generate_json.return_value=({'ok':True},{'latency_seconds':.1,'output_tokens':1})
            fake.model='fake'
            data=MagicMock();data.iter_records.return_value=[];data.__len__.return_value=0
            with patch.object(cli,'SYNTHESIS_ROOT',root),patch.object(cli,'OllamaClient',return_value=fake),patch.object(cli.ToolRegistry,'from_catalog',return_value=registry()),patch.object(cli.TrainOnlyDataset,'from_prepared_split',return_value=data),patch('sys.stdout',new_callable=io.StringIO):
                # Empty data is rejected for lack of eligible examples, not because output exists.
                data.iter_records.return_value=iter([record()])
                with patch('sys.argv',['tool','--dry-run','--output-dir',str(existing)]):cli.main()
                fake.generate_json.assert_not_called()
                with patch('sys.argv',['tool','--smoke','--output-dir',str(existing)]):cli.main()
                self.assertEqual(fake.generate_json.call_count,1)
                self.assertEqual(list(existing.iterdir()),[])
