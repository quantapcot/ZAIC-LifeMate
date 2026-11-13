import json
import unittest
from src.toolcall.agent.action_agent import ActionAgent
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.heldout import inputs_only
from src.toolcall.evaluation.metrics import metrics


def registry():
    schema={'type':'object','properties':{'id':{'type':'string'}},'required':['id']}
    return ToolRegistry({'apps':[{'id':'a','tools':[{'name':'a.read','parameters':{'type':'object','properties':{},'required':[]}},{'name':'a.write','parameters':schema}]},{'id':'sys','tools':[{'name':'sys.ask_missing_info','parameters':{'type':'object','properties':{'tool_name':{'type':'string'},'params':{'type':'array','items':{'type':'string'}}},'required':['tool_name','params']}}]}]})

class Model:
    def __init__(self, answers): self.answers=iter(answers); self.prompts=[]
    def generate_json(self, system, prompt, schema):
        self.prompts.append(json.loads(prompt)); return next(self.answers),{'output_tokens':1}

class Env:
    def __init__(self, reject=False): self.calls=[]; self.reject=reject; self.checks=[]
    def validate(self,n,a): self.checks.append((n,a)); return {'ok':not self.reject,'error':'rejected' if self.reject else None}
    def call(self,n,a): self.calls.append({'name':n,'arguments':a}); return {'ok':True,'id':'actual'}
    def get_tool_calls(self): return self.calls
    def get_state_hash(self): return 'official-hash'

call=lambda name,args: {'action':'call','name':name,'arguments':args}
finish={'action':'finish','name':'','arguments':{}}
context={'installed_apps':{'a':{'tools':['a.read','a.write']}}}

class AgentTests(unittest.TestCase):
    def test_observation_and_authoritative_output(self):
        model=Model([call('a.read',{}),call('a.write',{'id':'actual'}),finish]); env=Env()
        result=ActionAgent(model,registry()).run('xin chào',context,[],env)
        self.assertEqual(result['tool_calls'],env.calls); self.assertEqual(result['state_hash'],'official-hash')
        self.assertEqual(len(env.checks),2); self.assertEqual(model.prompts[1]['observations'][0]['result']['id'],'actual')
    def test_history_never_replayed(self):
        env=Env(); history=[{'role':'assistant','tool_calls':[{'name':'a.read','arguments':{}}]}]
        ActionAgent(Model([finish]),registry()).run('hi',context,history,env)
        self.assertEqual(env.calls,[])
    def test_unavailable_and_missing_argument_not_executed(self):
        env=Env(); model=Model([call('b.bad',{}),call('a.write',{}),finish])
        result=ActionAgent(model,registry()).run('hi',context,[],env)
        self.assertEqual(env.calls,[]); self.assertEqual(result['invalid_proposals'],2)
    def test_sdk_rejection_not_executed(self):
        env=Env(True); result=ActionAgent(Model([call('a.read',{}),finish]),registry()).run('hi',context,[],env)
        self.assertEqual(env.calls,[]); self.assertEqual(result['sdk_checks'],1); self.assertEqual(result['sdk_valid'],0)
    def test_repeat_stops(self):
        env=Env(); result=ActionAgent(Model([call('a.read',{}),call('a.read',{})]),registry()).run('hi',context,[],env)
        self.assertEqual(len(env.calls),1); self.assertEqual(result['error'],'repeated_call')
    def test_missing_info(self):
        env=Env(); result=ActionAgent(Model([call('sys.ask_missing_info',{'tool_name':'a.write','params':['id']})]),registry()).run('hi',context,[],env)
        self.assertEqual(len(result['tool_calls']),1)
    def test_empty_context_fails_closed(self):
        env=Env(); ActionAgent(Model([call('a.read',{}),finish]),registry()).run('hi',{},[],env); self.assertEqual(env.calls,[])
    def test_labels_are_skipped_even_with_escaped_braces(self):
        row=inputs_only('{"id":"x","label":{"text":"}\\\"{","nested":[{}]},"audio_text":"xin","context":{},"history":[]}')
        self.assertNotIn('label',row); self.assertEqual(row['audio_text'],'xin')
    def test_metrics_duplicates_and_empty(self):
        def row(i,c): return {'id':i,'tool_calls':c,'latency_seconds':1,'model_seconds':1,'tool_seconds':0,'output_tokens':1,'invalid_proposals':0,'sdk_valid':0,'sdk_checks':0,'execution_success':0,'executions':0,'error':None}
        c={'name':'a.read','arguments':{}}; p=[row('a',[c,c]),row('b',[])]; labels={'a':{'tool_calls':[c]},'b':{'tool_calls':[]}}
        m=metrics(p,labels); self.assertEqual(m['call_precision'],.5); self.assertEqual(m['call_recall'],1); self.assertEqual(m['call_sequence_exact_match'],.5)
        self.assertIsNone(m['official_score']); self.assertIsNone(m['final_state_hash_match'])

if __name__=='__main__': unittest.main()
