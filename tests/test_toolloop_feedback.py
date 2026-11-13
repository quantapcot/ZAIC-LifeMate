import copy
import json
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock
from src.toolcall.synthesis.refinement import run_stage
from src.toolcall.synthesis import pipeline, validators, sampler
from src.toolcall.verification.semantic_verifier import inspect_verdict, SemanticVerifier
from src.toolcall.verification.policies import REQUIRED
from src.toolcall.llm import OllamaClient, TransportError, InvalidModelOutput
from tests.test_toolloop_synthesis import registry, record, ground, FakeClient, Response


def verdict(stage,status='pass'):
    return {'stage':stage,'verdict':status,'checks':[{'name':n,'status':'pass','field':'','error_code':'','message':'','source':'','text_span':''} for n in REQUIRED[stage]],'summary':'fixture'}


def provenance(path, span, source='user_query'):
    return {'name':'argument_provenance','status':'pass','field':path,'error_code':'','message':'','source':source,'text_span':span}


class VerdictTests(unittest.TestCase):
    def test_required_rubric_fail_and_uncertain(self):
        for stage in (1,2,3):
            v=verdict(stage);self.assertEqual(inspect_verdict(stage,v,registry=registry()),[])
            bad=copy.deepcopy(v);bad['checks'].pop();self.assertTrue(inspect_verdict(stage,bad,registry=registry()))
            bad=copy.deepcopy(v);bad['checks'][0]['status']='fail';bad['verdict']='fail';self.assertTrue(inspect_verdict(stage,bad,registry=registry()))
            bad=copy.deepcopy(v);bad['verdict']='uncertain';self.assertTrue(inspect_verdict(stage,bad,registry=registry()))
            with self.assertRaises(ValueError):inspect_verdict(stage,{'stage':stage},registry=registry())

    def test_argument_provenance_must_match_query_and_value(self):
        c=[{'name':'demo.run','arguments':{'x':'a'}}];v=verdict(3);v['checks'].append(provenance('tool_calls[0].arguments.x','a'))
        self.assertEqual(inspect_verdict(3,v,query='value a',calls=c,registry=registry()),[])
        bad=copy.deepcopy(v);bad['checks'][-1]['text_span']='fabricated'
        self.assertIn('UNSUPPORTED_ARGUMENT_VALUE',inspect_verdict(3,bad,query='value a',calls=c,registry=registry()))
        bad=copy.deepcopy(v);bad['checks'].pop();self.assertIn('MISSING_ARGUMENT_PROVENANCE',inspect_verdict(3,bad,query='value a',calls=c,registry=registry()))
        bad=copy.deepcopy(v);bad['checks'][-1]['source']='schema_default'
        self.assertIn('INVALID_SCHEMA_DEFAULT',inspect_verdict(3,bad,query='value a',calls=c,registry=registry()))

    def test_model_verdict_cannot_hide_missing_or_failed_checks(self):
        for item in ('intent_consistency','required_parameter_coverage','no_invented_state'):
            v=verdict(2);next(c for c in v['checks'] if c['name']==item)['status']='fail'
            self.assertTrue(inspect_verdict(2,v))
        v=verdict(3);v['checks'][0]['status']='not_verified'
        self.assertTrue(inspect_verdict(3,v,registry=registry()))

    def test_semantic_client_malformed_and_timeout(self):
        c=MagicMock();c.generate_json.side_effect=InvalidModelOutput('bad',code='INVALID_JSON')
        with self.assertRaises(InvalidModelOutput):SemanticVerifier(c).verify(1,ground(),registry())
        c.generate_json.side_effect=TransportError('timeout',code='TIMEOUT')
        with self.assertRaises(TransportError):SemanticVerifier(c).verify(1,ground(),registry())
        with self.assertRaises(ValueError):SemanticVerifier(None)


class RefinementTests(unittest.TestCase):
    def test_zero_to_three_repairs_and_retry_cap(self):
        for failures in (0,1,2,3,4):
            produced=[]
            def producer(attempt,instruction):
                produced.append((attempt,instruction));return {'attempt':attempt},{'latency_seconds':0.01}
            def deterministic(output):
                return validators.feedback(2, [('field','invalid_type')] if output['attempt']<failures else [])
            def semantic(output):return {'passed':True,'issues':[],'verdict':verdict(2)}
            r=run_stage(2,producer,deterministic,semantic,3)
            self.assertEqual(r['status'],'passed' if failures<4 else 'rejected')
            self.assertEqual(len(produced),min(failures+1,4))
            self.assertTrue(all(x['semantic'] is None for x in r['history'][:-1]))
            if failures: self.assertIn('previous_output',produced[1][1])
        with self.assertRaises(ValueError):run_stage(2,lambda *x:None,lambda x:None,None)
        with self.assertRaises(ValueError):run_stage(2,lambda *x:None,lambda x:None,lambda x:None,4)

    def test_semantic_fail_is_not_overridden_and_uncertain_manual(self):
        def producer(a,i):return {'x':1},{}
        def check(x):return validators.feedback(2,[])
        def rejected(x):return {'passed':False,'issues':['QUERY_TOOL_MISMATCH'],'verdict':verdict(2,'fail')}
        r=run_stage(2,producer,check,rejected,1)
        self.assertEqual(r['status'],'rejected');self.assertEqual(len(r['history']),2)
        def uncertain(x):return {'passed':False,'issues':['UNCERTAIN'],'verdict':verdict(2,'uncertain')}
        self.assertEqual(run_stage(2,producer,check,uncertain,3)['status'],'manual_review_required')

    def test_refinement_redacts_expected_tool_from_feedback(self):
        prompts=[]
        def producer(attempt,instruction):
            prompts.append(instruction)
            return {'n':attempt},{}
        def check(output):return validators.feedback(3,[])
        def semantic(output):
            v=verdict(3,'fail' if output['n']==0 else 'pass')
            if output['n']==0:
                v['checks'][0]['status']='fail';v['checks'][0]['message']='Use demo.run for this intent.'
                return {'passed':False,'issues':['MISMATCH'],'verdict':v}
            return {'passed':True,'issues':[],'verdict':v}
        result=run_stage(3,producer,check,semantic,1,redact_terms=['demo.run'])
        self.assertEqual(result['status'],'passed')
        self.assertNotIn('demo.run',json.dumps(prompts[1]['issues']))
        self.assertIn('demo.run',result['history'][0]['semantic']['verdict']['checks'][0]['message'])

    def test_transport_is_not_refined(self):
        def producer(a,i):raise TransportError('unavailable',code='TIMEOUT')
        r=run_stage(2,producer,lambda x:None,lambda x:None,3)
        self.assertEqual(len(r['history']),1);self.assertEqual(r['history'][0]['issues'][0]['source'],'transport')

    def test_existing_pipeline_verified_chain_and_no_stale_downstream(self):
        class Verifier:
            def __init__(self): self.client=type('C',(),{'model':'independent'})();self.stages=[]
            def verify(self,stage,ground,registry,query='',calls=None):
                self.stages.append(stage)
                result=verdict(stage)
                for i,call in enumerate(calls or []):
                    for key,value in call['arguments'].items():
                        result['checks'].append(provenance(f'tool_calls[{i}].arguments.{key}',str(value)))
                return {'passed':True,'issues':[],'verdict':result,'usage':{'latency_seconds':0.01}}
        meta={'generator':'g','generator_digest':'123','verifier':'v','verifier_digest':'456'}
        v=Verifier();out=pipeline.run_verified_attempt(record(),registry(),FakeClient(),'simple',42,set(),v,0,meta)
        self.assertEqual(out['status'],'verified_candidate');self.assertFalse(out['sft_ready']);self.assertEqual(v.stages,[1,2,3])
        self.assertEqual(set(out['stage_history']),{'ground_truth','backward_query','forward_calls'})
        class BadVerifier(Verifier):
            def verify(self,stage,*a,**kw):
                self.stages.append(stage);return {'passed':False,'issues':['failure'],'verdict':verdict(stage,'uncertain')}
        v=BadVerifier();out=pipeline.run_verified_attempt(record(),registry(),FakeClient(),'simple',42,set(),v,0,meta)
        self.assertEqual(out['status'],'manual_review_required');self.assertEqual(v.stages,[1]);self.assertNotIn('forward_calls',out['stages'])

    def test_refined_query_is_only_input_to_later_stage(self):
        class Gen:
            model='generator'
            def __init__(self): self.inputs=[]
            def generate_json(self,system_prompt,user_prompt,output_schema):
                payload=json.loads(user_prompt);self.inputs.append(payload)
                if 'targets' in payload:
                    return {'user_query':'old a' if len(self.inputs)==1 else 'revised a'},{'latency_seconds':0.01}
                return {'tool_calls':[{'name':'demo.run','arguments':{'x':'a'}}]},{'latency_seconds':0.01}
        class Verify:
            def __init__(self):self.client=type('C',(),{'model':'independent'})();self.query_checks=0
            def verify(self,stage,ground,registry,query='',calls=None):
                v=verdict(stage)
                if stage==2:
                    self.query_checks+=1
                    if self.query_checks==1:
                        v['verdict']='fail';v['checks'][0]['status']='fail';v['checks'][0]['error_code']='UNNATURAL_VIETNAMESE';v['checks'][0]['message']='revise'
                        return {'verdict':v,'issues':['UNNATURAL_VIETNAMESE'],'passed':False}
                if stage==3:v['checks'].append(provenance('tool_calls[0].arguments.x','a'))
                return {'verdict':v,'issues':[],'passed':True}
        meta={'generator':'g','generator_digest':'123','verifier':'v','verifier_digest':'456'}
        client=Gen();out=pipeline.run_verified_attempt(record(),registry(),client,'simple',42,set(),Verify(),3,meta)
        self.assertEqual(out['status'],'verified_candidate')
        self.assertEqual(client.inputs[-1]['query'],'revised a')
        self.assertIn('refinement',client.inputs[1])
        self.assertEqual(len(out['stage_history']['backward_query']),2)

    def test_same_digest_and_missing_verifier_fail_closed(self):
        from src.toolcall.data import TrainOnlyDataset
        view=MagicMock(spec=TrainOnlyDataset);view.iter_records.return_value=iter([record()])
        client=MagicMock();client.model='generator';client.model_digest.return_value='same';client.options={'num_ctx':4096,'num_predict':512}
        verifier=MagicMock();verifier.client.model='different';verifier.client.model_digest.return_value='same';verifier.client.options={'num_ctx':4096,'num_predict':512}
        with self.assertRaises(ValueError):pipeline.generate(view,registry(),client,1,42,verifier=verifier)
        with self.assertRaises(ValueError):pipeline.run_verified_attempt(record(),registry(),client,'simple',42,set(),None)

    def test_deterministic_failure_prevents_verifier_call(self):
        verify=MagicMock()
        outcome=run_stage(3,lambda a,i:({'tool_calls':[]},{}),lambda o:validators.feedback(3,[('arguments','missing_required')]),verify,0)
        self.assertEqual(outcome['status'],'rejected')
        verify.assert_not_called()

    def test_model_settings_and_prompt_context_are_minimal(self):
        c=OllamaClient(keep_alive=0,think=False);c.opener=MagicMock()
        c.opener.open.return_value=Response(json.dumps({'done':True,'message':{'content':'{}'}}).encode())
        c.generate_json('system','user',{'type':'object'})
        payload=json.loads(c.opener.open.call_args.args[0].data)
        self.assertEqual(payload['keep_alive'],0);self.assertFalse(payload['think'])
        fake=MagicMock();fake.generate_json.return_value=(verdict(1),{'latency_seconds':0.01})
        SemanticVerifier(fake).verify(1,ground(),registry())
        sent=json.loads(fake.generate_json.call_args.args[1])
        self.assertNotIn('history',sent);self.assertNotIn('user_profile',sent)
        self.assertNotIn('audio_text',sent)

    def test_model_unload_local_and_oversize(self):
        client=OllamaClient(keep_alive=0,think=False);client.opener=MagicMock();client.opener.open.return_value=Response(b'{}')
        client.unload();req=client.opener.open.call_args.args[0]
        self.assertEqual(json.loads(req.data)['keep_alive'],0)
        client.opener.open.return_value=Response(b'x'*(4*1024*1024+1))
        with self.assertRaises(InvalidModelOutput) as result:client.generate_json('s','u',{'type':'object'})
        self.assertEqual(result.exception.code,'RESPONSE_TOO_LARGE')

    def test_oom_is_technical_and_not_retried(self):
        import io
        from urllib.error import HTTPError
        c=OllamaClient();c.opener=MagicMock()
        c.opener.open.side_effect=HTTPError('http://127.0.0.1',500,'error',{},io.BytesIO(b'CUDA out of memory'))
        with self.assertRaises(TransportError) as error:
            c.generate_json('s','u',{'type':'object'})
        self.assertEqual(error.exception.code,'OUT_OF_MEMORY')
        c.opener.open.assert_called_once()

    def test_unsupported_distractors_filtered(self):
        r=registry();r._tools['demo.other']['parameters']['properties']['state']={'type':'string','x-ref':'foo'}
        with self.assertRaisesRegex(ValueError,'no_coherent_distractor'):sampler.sample(record(),r,'multiple',42)
