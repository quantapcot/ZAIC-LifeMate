import unittest
from src.toolcall.agent.action_agent import ActionAgent,available_tools
from src.toolcall.agent.tool_router import ToolRouter,tokens
from src.toolcall.evaluation.routing_diagnostics import aligned_recall,recall_at_k
from scripts.diagnose_action_selection import oracle_candidates
from tests.test_action_agent_c1 import Model,Env,registry,context,call,finish

class C12Tests(unittest.TestCase):
    def test_name_repetition_separator(self):
        name='demo.run';self.assertEqual(tokens(' '.join([name,name])),['demo','run','demo','run'])
        r=registry();r._tools['a.read']['description']='';r._tools['a.write']['description']=''
        self.assertEqual(ToolRouter(r,1).rank(['a.read','a.write'],'write')[0][0],'a.write')
    def test_normalization_and_determinism(self):
        r=registry();r._tools['a.read']['description']='Tìm thiết bị'
        router=ToolRouter(r,1)
        self.assertEqual(router.rank(['a.read','a.write'],'tim thiet bi'),router.rank(['a.read','a.write'],'tìm thiết bị'))
        self.assertEqual(router.rank(['a.write','a.read'],'zzz')[0][0],'a.read')
    def test_only_installed(self):
        r=registry();allowed=available_tools({'installed_apps':{'a':{'tools':['a.read']}}},r)
        self.assertNotIn('a.write',ToolRouter(r,8).rank(allowed,'write')[0])
    def test_recall_and_step_alignment(self):
        self.assertEqual(recall_at_k(['a','b'],['b','c'],1),(0,2))
        gold=[{'name':'a','arguments':{}},{'name':'b','arguments':{}}]
        trace=[{'candidate_tools':['a'],'executed':True,'proposed_tool':'a','proposed_arguments':{}},{'candidate_tools':['c'],'executed':False}]
        self.assertEqual(aligned_recall(trace,gold),(1,2))
        # Tool b appearing before it is needed must not count for its later step.
        trace[0]['candidate_tools'].append('b');self.assertEqual(aligned_recall(trace,gold),(1,2))
    def test_oracle_rejects_nontrain(self):
        for p in ('val','test','public','private'):
            with self.assertRaises(ValueError):oracle_candidates(p,['a'],['a'],'a',42)
        self.assertIn('b',oracle_candidates('train',['a','b'],['a'],'b',42))
        with self.assertRaises(ValueError):oracle_candidates('train',['a'],['a'],{'name':'a','arguments':{'secret':1}},42)
    def test_progress_recheck_then_dependent_action(self):
        model=Model([call('a.read',{}),finish,call('a.write',{'id':'actual'}),finish]);env=Env()
        result=ActionAgent(model,registry()).run('find then write',context,[],env)
        self.assertEqual(len(env.calls),2);self.assertEqual(result['trace'][1]['stop_reason'],'PROGRESS_RECHECK')
        self.assertEqual(model.prompts[2]['observations'][0]['result']['id'],'actual')
    def test_out_of_scope_empty(self):
        env=Env();result=ActionAgent(Model([finish]),registry()).run('unknown',context,[],env)
        self.assertEqual(result['tool_calls'],[])
    def test_failed_prerequisite_not_fabricated(self):
        class Failed(Env):
            def call(self,n,a):self.calls.append({'name':n,'arguments':a});return {'ok':False,'error':'missing'}
        model=Model([call('a.read',{}),finish,finish]);result=ActionAgent(model,registry()).run('x',context,[],Failed())
        self.assertEqual(result['trace'][1]['stop_reason'],'PROGRESS_RECHECK');self.assertEqual(len(result['tool_calls']),1)

if __name__=='__main__':unittest.main()
