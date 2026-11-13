import json
import unittest
from src.toolcall.agent.action_agent import ActionAgent, basic_issues, compact_tool
from src.toolcall.agent.tool_router import ToolRouter
from src.toolcall.llm.ollama_client import PromptBudgetError
from src.toolcall.evaluation.metrics import metrics
from tests.test_action_agent_c1 import Model, Env, registry, context, call, finish

class RoutingTests(unittest.TestCase):
    def test_metadata_descriptions_choose_relevant_installed_tool(self):
        r=registry(); r._tools['a.read']['description']='Tìm thiết bị theo tên';r._tools['a.write']['description']='Gửi tin nhắn'
        candidates,scores=ToolRouter(r,1).rank(['a.read','a.write','sys.ask_missing_info'],'tìm thiết bị',{})
        self.assertIn('a.read',candidates); self.assertNotIn('a.write',candidates); self.assertIn('sys.ask_missing_info',candidates)
    def test_preserves_nested_semantics(self):
        tool={'name':'a.test','description':'full description','parameters':{'type':'object','required':['id'],'properties':{'id':{'type':'string','description':'device identifier','enum':['x'],'x-ref':'device'}}}}
        self.assertEqual(compact_tool(tool),tool)
    def test_unknown_constraints_delegated(self):
        self.assertEqual(basic_issues({'id':'x'},{'type':'object','properties':{'id':{'type':'string','x-ref':'device','format':'uuid'}},'required':['id']}),[])
    def test_nested_type_and_required(self):
        s={'type':'object','required':['nested'],'properties':{'nested':{'type':'object','required':['x'],'properties':{'x':{'type':'integer'}}}}}
        self.assertEqual(basic_issues({'nested':{'x':'bad'}},s)[0]['code'],'INVALID_ARGUMENT_TYPE')
        self.assertEqual(basic_issues({'nested':{}},s)[0]['code'],'MISSING_REQUIRED_ARGUMENT')
    def test_repeat_after_intervening_call_allowed(self):
        model=Model([call('a.read',{}),call('a.write',{'id':'actual'}),call('a.read',{}),finish]);env=Env()
        result=ActionAgent(model,registry()).run('x',context,[],env)
        self.assertEqual(len(result['tool_calls']),3)
    def test_fallback_shortlist_then_finish(self):
        model=Model([finish,finish]);result=ActionAgent(model,registry(),top_k=1).run('unknown',context,[],Env())
        self.assertEqual(len(model.prompts),2);self.assertEqual(result['trace'][0]['stop_reason'],'FALLBACK_RETRIEVAL')
    def test_prompt_budget_failure_is_distinct(self):
        class TooLarge:
            def generate_json(self,*args): raise PromptBudgetError({'estimated_input_tokens':99999})
        result=ActionAgent(TooLarge(),registry()).run('x',context,[],Env())
        self.assertEqual(result['error'],'prompt_budget');self.assertEqual(result['trace'][-1]['stop_reason'],'PROMPT_BUDGET_EXCEEDED')
    def test_rejection_feedback_is_actionable(self):
        model=Model([call('a.write',{}),call('a.write',{'id':'actual'}),finish]);env=Env()
        result=ActionAgent(model,registry()).run('x',context,[],env)
        self.assertEqual(result['trace'][0]['internal_validation'],'rejected')
        self.assertEqual(model.prompts[1]['observations'][0]['rejected_tool'],'a.write')
        self.assertEqual(len(env.calls),1)
    def test_sdk_rejects_enum_without_execution(self):
        r=registry();r._tools['a.write']['parameters']['properties']['id']['enum']=['allowed']
        env=Env(reject=True);result=ActionAgent(Model([call('a.write',{'id':'other'}),finish]),r).run('x',context,[],env)
        self.assertEqual(result['trace'][0]['internal_validation'],'pass')
        self.assertEqual(result['trace'][0]['sdk_validation'],'rejected');self.assertFalse(env.calls)
    def test_step_limit(self):
        env=Env();result=ActionAgent(Model([call('a.read',{})]),registry(),max_steps=1).run('x',context,[],env)
        self.assertEqual(result['error'],'step_limit');self.assertEqual(result['trace'][-1]['stop_reason'],'STEP_LIMIT')
    def test_model_input_excludes_unrelated_context(self):
        model=Model([finish]);ctx=dict(context,label={'secret':'gold'},final_state={'x':'hidden'})
        ActionAgent(model,registry()).run('hello',ctx,[],Env())
        prompt=json.dumps(model.prompts[0]);self.assertNotIn('gold',prompt);self.assertNotIn('hidden',prompt)
    def test_router_uses_new_observation(self):
        r=registry();r._tools['a.read']['description']='find devices';r._tools['a.write']['description']='turn lights on'
        router=ToolRouter(r,1)
        before,_=router.rank(['a.read','a.write'],'find devices')
        after,_=router.rank(['a.read','a.write'],'',observations=[{'result':'turn lights on'}])
        self.assertEqual(before[0],'a.read');self.assertEqual(after[0],'a.write')
    def test_offline_candidate_recall(self):
        c={'name':'a.read','arguments':{}}
        row={'id':'x','tool_calls':[],'latency_seconds':1,'model_seconds':1,'tool_seconds':0,'output_tokens':0,'invalid_proposals':0,'sdk_valid':0,'sdk_checks':0,'execution_success':0,'executions':0,'error':None,'trace':[{'candidate_tools':['a.read'],'candidate_count':1,'model_action':'finish','stop_reason':'FINISH'}]}
        m=metrics([row],{'x':{'tool_calls':[c]}})
        self.assertEqual(m['candidate_recall_union'],1); self.assertEqual(m['premature_finish_by_count'],1)

if __name__=='__main__': unittest.main()
