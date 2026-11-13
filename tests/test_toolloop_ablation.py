import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
from src.toolcall.data import TrainOnlyDataset
from src.toolcall.llm import OllamaClient, TransportError
from src.toolcall.synthesis import pipeline, sampler
from src.toolcall.verification.policies import REQUIRED
from scripts import ablate_toolloop_feedback as ablation
from tests.test_toolloop_synthesis import registry,record,Response,FakeClient
from tests.test_toolloop_feedback import verdict,provenance


class PromptBudgetTests(unittest.TestCase):
    def client(self, **kwargs):
        c=OllamaClient(**kwargs);c.opener=MagicMock();c.opener.open.return_value=Response(b'{"done":true,"message":{"content":"{}"}}');return c

    def test_unicode_nested_schema_is_not_byte_token_equivalence(self):
        c=self.client();text='\u0111\u1eb7t l\u1ecbch ' * 250
        schema={'type':'object','properties':{'nested':{'type':'object','properties':{'note':{'type':'string','description':text}}}}}
        _,usage=c.generate_json('s','small Vietnamese',schema)
        self.assertGreater(usage['input_budget']['payload_bytes'],3328)
        self.assertLess(usage['input_budget']['estimated_input_tokens'],4096-512)
        c.opener.open.assert_called_once()

    def test_unload_confirmation_checks_digest_and_stops(self):
        c=self.client(confirm_unloaded=True);c._model_digest='same'
        c.opener.open.return_value=Response(json.dumps({'models':[{'name':'alias','digest':'same'}]}).encode())
        with patch('src.toolcall.llm.ollama_client.time.sleep'):
            with self.assertRaises(TransportError) as error:c.assert_unloaded()
        self.assertEqual(error.exception.code,'UNLOAD_UNCONFIRMED')
        c.opener.open.return_value=Response(b'{"models":[]}');c.assert_unloaded()

    def test_long_small_oversized_and_context_reservation(self):
        c=self.client();c.generate_json('s','x'*2000,{'type':'object'});self.assertGreater(c.last_budget['estimated_input_tokens'],2000)
        c=self.client(context_length=1024,max_tokens=128)
        with self.assertRaisesRegex(ValueError,'PROMPT_TOO_LARGE') as error:c.generate_json('s','x'*1000,{'type':'object'})
        self.assertEqual(error.exception.budget['reserved_output_tokens'],128)
        c.opener.open.assert_not_called();self.assertEqual(c.last_budget['reserved_output_tokens'],128)
        c=self.client(context_length=8192,max_tokens=256);c.generate_json('s','x'*4000,{'type':'object'})
        with self.assertRaises(ValueError):OllamaClient(context_length=512,max_tokens=400)
        c=self.client(max_payload_bytes=1024)
        with self.assertRaises(ValueError):c.generate_json('s','x'*2000,{'type':'object'})
        c.opener.open.assert_not_called()


def richer_registry(count=3):
    r=registry()
    for i in range(count):r._tools['demo.extra'+str(i)]={'name':'demo.extra'+str(i),'kind':'read','parameters':{'type':'object','properties':{},'required':[]}};r._groups['demo.extra'+str(i)]=('demo','example')
    return r


def rich_record(count=3):
    r=record();r['context']['installed_apps']['demo']['tools'] += ['demo.extra'+str(i) for i in range(count)];return r


class GroundCorrectionTests(unittest.TestCase):
    def run_ground(self, scenario, reg, rec, always=False):
        class V:
            def __init__(self):self.stage1=0;self.client=type('X',(),{'model':'v'})()
            def verify(self,stage,ground,registry,query='',calls=None):
                v=verdict(stage)
                if stage==1:
                    self.stage1+=1
                    if always or self.stage1==1:
                        v['verdict']='fail';c=next(c for c in v['checks'] if c['name']=='distractor_relevance');c['status']='fail';c['error_code']='INVALID_DISTRACTOR'
                        return {'verdict':v,'issues':['INVALID_DISTRACTOR'],'passed':False}
                if stage==3:
                    for i,call in enumerate(calls or []):
                        for key,value in call['arguments'].items():v['checks'].append(provenance(f'tool_calls[{i}].arguments.{key}',str(value)))
                return {'verdict':v,'issues':[],'passed':True}
        meta={'generator':'g','generator_digest':'1','verifier':'v','verifier_digest':'2'}
        return pipeline.run_verified_attempt(rec,reg,FakeClient(),scenario,42,set(),V(),3,meta)

    def test_simple_rejection_does_not_repeat(self):
        result=self.run_ground('simple',registry(),record())
        h=result['stage_history']['ground_truth'];self.assertEqual(len(h),1);self.assertEqual(h[0]['stop_reason'],'invariant_simple_target')

    def test_multiple_target_infeasibility_stops_early(self):
        class V:
            client=type('X',(),{'model':'v'})()
            def verify(self,stage,ground,registry,query='',calls=None):
                v=verdict(stage,'fail');c=next(c for c in v['checks'] if c['name']=='feasibility');c['status']='fail';c['error_code']='TARGET_REQUIRES_UNKNOWN_CONTEXT'
                return {'verdict':v,'issues':['TARGET_REQUIRES_UNKNOWN_CONTEXT'],'passed':False}
        meta={'generator':'g','generator_digest':'1','verifier':'v','verifier_digest':'2'}
        r=pipeline.run_verified_attempt(rich_record(),richer_registry(),FakeClient(),'multiple',42,set(),V(),3,meta)
        h=r['stage_history']['ground_truth']
        self.assertEqual(len(h),1);self.assertEqual(h[0]['stop_reason'],'feedback_not_correctable_by_distractor_change')

    def test_distinct_alternatives_and_cap(self):
        r=rich_record();reg=richer_registry();x=self.run_ground('multiple',reg,r,True)
        h=x['stage_history']['ground_truth'];self.assertEqual(len(h),4)
        signatures=[tuple(sorted(a['output']['candidate_tools'])) for a in h]
        self.assertEqual(len(set(signatures)),4)
        again=self.run_ground('multiple',reg,r,True)
        self.assertEqual(signatures,[tuple(sorted(a['output']['candidate_tools'])) for a in again['stage_history']['ground_truth']])
        y=self.run_ground('multiple',reg,r);self.assertEqual(len(y['stage_history']['ground_truth']),2)
        z=self.run_ground('multiple',registry(),record(),True);self.assertEqual(len(z['stage_history']['ground_truth']),1);self.assertEqual(z['stage_history']['ground_truth'][0]['stop_reason'],'no_alternative_distractor')


class Generator:
    model='fake-generator';options={'num_ctx':4096,'num_predict':512};think=None
    def __init__(self):self.calls=[];self.keep_alive=0;self.confirm_unloaded=False
    def model_digest(self):return 'generator-digest'
    def unload(self):pass
    def generate_json(self,system_prompt,user_prompt,output_schema):
        payload=json.loads(user_prompt);self.calls.append(payload)
        usage={'latency_seconds':.01,'prompt_tokens':5,'output_tokens':3}
        if 'targets' in payload:
            return {'user_query':'revised value a' if 'refinement' in payload else 'initial value a'},usage
        return {'tool_calls':[{'name':'demo.run','arguments':{'x':'a'}}]},usage


class Verifier:
    def __init__(self,repair=False):
        self.client=type('C',(),{'model':'fake-verifier','options':{'num_ctx':4096,'num_predict':512},'think':False,'model_digest':lambda s:'verifier-digest','unload':lambda s:None})()
        self.repair=repair;self.stage2=0
    def verify(self,stage,ground,registry,query='',calls=None):
        v=verdict(stage)
        if stage==2:
            self.stage2+=1
            # In paired B/C order, the second verifier visit is C's frozen initial query.
            if self.repair and self.stage2==2:
                v['verdict']='fail';v['checks'][0]['status']='fail';v['checks'][0]['error_code']='UNNATURAL_VIETNAMESE';v['checks'][0]['message']='revise'
                return {'verdict':v,'issues':['UNNATURAL_VIETNAMESE'],'passed':False,'usage':{'latency_seconds':.01,'output_tokens':1}}
        if stage==3:
            for i,call in enumerate(calls or []):
                for key,value in call['arguments'].items():v['checks'].append(provenance(f'tool_calls[{i}].arguments.{key}',str(value)))
        return {'verdict':v,'issues':[],'passed':True,'usage':{'latency_seconds':.01,'output_tokens':1}}


class SyntheticVerifierControlsTests(unittest.TestCase):
    def test_controls_do_not_conflate_policy_rejection_with_verifier_detection(self):
        from scripts.smoke_toolloop_verifier import run_controls
        client=MagicMock()
        usage={'latency_seconds':0.01}
        client.generate_json.side_effect=[({'ok':True},usage),(verdict(2),usage),(verdict(2,'fail'),usage),(verdict(3,'pass'),usage),(verdict(2,'uncertain'),usage)]
        results=run_controls(client)
        self.assertFalse(results['all_controls_met'])
        self.assertFalse(results['controls']['hallucinated_argument']['policy_passed'])
        self.assertFalse(results['controls']['hallucinated_argument']['met_expectation'])



class PairedAblationTests(unittest.TestCase):
    def view(self,n=2):
        rows=[]
        for i in range(n):r=record();r['id']=str(i);rows.append(r)
        view=MagicMock(spec=TrainOnlyDataset);view.iter_records.side_effect=lambda:iter(rows);return view

    def test_shared_frozen_baseline_and_no_resampling(self):
        gen=Generator();art=ablation.paired(self.view(),registry(),gen,Verifier(),2,42,3)
        self.assertEqual(len(gen.calls),4)
        for i in range(2):
            a=art['initial_candidates.json'][i]
            self.assertEqual(a,art['results_A.json'][i])
            self.assertEqual(a['scenario_id'],art['experiment_manifest.json']['scenarios'][i]['id'])
            for mode in ('results_B.json','results_C.json'):
                item=art[mode][i]
                self.assertEqual(item['initial_candidate_sha256'],ablation.digest_json(a))
                self.assertEqual(item['stage_history']['ground_truth'][0]['output'],a['stages']['ground_truth']['output'])
                self.assertEqual(item['stage_history']['backward_query'][0]['output'],a['stages']['backward_query']['model_output'])
                self.assertEqual(item['stage_history']['forward_calls'][0]['output'],a['stages']['forward_calls']['model_output'])
                self.assertTrue(item['stage_history']['forward_calls'][0]['generation_usage']['reused_initial'])
            self.assertEqual(art['experiment_manifest.json']['scenarios'][i]['seed'],42+i)
        self.assertTrue(all(x['status']=='not_reviewed' for x in art['human_review.json']))

    def test_upstream_revision_regenerates_downstream_without_target_leak(self):
        gen=Generator();art=ablation.paired(self.view(1),registry(),gen,Verifier(repair=True),1,42,3)
        c=art['results_C.json'][0]
        self.assertEqual(c['status'],'verified_candidate')
        self.assertEqual(len(c['stage_history']['backward_query']),2)
        self.assertNotIn('reused_initial',c['stage_history']['forward_calls'][0]['generation_usage'])
        forward=[x for x in gen.calls if 'candidate_tools' in x]
        self.assertEqual(forward[-1]['query'],'revised value a');self.assertNotIn('targets',forward[-1])

    def test_safe_local_no_overwrite_and_oom_stop(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d).resolve()/'ablation'
            with patch.object(ablation,'ROOT',base):
                out=base/'run';ablation.write_artifacts(out,{'summary.json':{'outcome':'partial'}})
                self.assertTrue((out/'PARTIAL').exists());self.assertFalse((out/'COMPLETE').exists())
                with patch('scripts.ablate_toolloop_feedback.os.replace',side_effect=OSError('marker failed')):
                    with self.assertRaises(OSError):ablation.write_artifacts(base/'interrupted',{'summary.json':{'outcome':'completed'}})
                self.assertFalse((base/'interrupted'/'COMPLETE').exists())
                with self.assertRaises(FileExistsError):ablation.write_artifacts(out,{})
                with self.assertRaises(ValueError):ablation.write_artifacts(base.parent/'escape',{})
        gen=Generator()
        def fail(*args,**kwargs):raise TransportError('oom',code='OUT_OF_MEMORY')
        gen.generate_json=MagicMock(side_effect=fail)
        result=ablation.paired(self.view(),registry(),gen,Verifier(),2,42,3)
        self.assertEqual(result['summary.json']['outcome'],'partial');self.assertEqual(len(result['initial_candidates.json']),1)
        gen.generate_json.assert_called_once()
