import json
import unittest
import importlib.util
import tempfile
import random
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path
from src.toolcall.training.dataset_builder import first_target,partition,prepare_prompt,render
from src.toolcall.training.transformers_backend import TransformersActionModel
from scripts.evaluate_action_sft import summarize,compact_routed
from tests.test_action_agent_c1 import registry,context

class ToyTokenizer:
    eos_token_id=999
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        text='|'.join(m['role']+':'+m['content'] for m in messages)
        if add_generation_prompt:text+='|assistant:'
        result=[ord(c) for c in text]
        if not add_generation_prompt:result += [999,10]
        return result

class SftTests(unittest.TestCase):
    def test_first_target_and_finish(self):
        self.assertEqual(first_target([]),{'action':'finish','name':'','arguments':{}})
        call={'name':'a.write','arguments':{'id':'actual'}}
        self.assertEqual(first_target([call])['arguments'],{'id':'actual'})
    def test_group_split_stable(self):
        self.assertEqual(partition('Hello  WORLD'),partition(' hello world '))
    def test_training_gold_candidate_with_schema_and_no_results(self):
        r=registry();target=first_target([{'name':'a.write','arguments':{'id':'actual'}}]);row={'id':'train_x','audio_text':'read','context':context,'history':[],'label':{'secret':'never'}}
        payload=prepare_prompt(row,r,target,top_k=1)
        names=[x['name'] for x in payload['tools']];self.assertIn('a.write',names);self.assertEqual(payload['observations'],[])
        self.assertNotIn('secret',json.dumps(payload));self.assertIn('id',next(x for x in payload['tools'] if x['name']=='a.write')['parameters']['properties'])
    def test_dataset_builder_does_not_accept_validation_partition(self):
        result=subprocess.run([sys.executable,'-m','scripts.prepare_action_sft','--output-dir','outputs/action_sft/test-unused','--split','val'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('unrecognized arguments',result.stderr)
    def test_test_partition_is_refused(self):
        from src.toolcall.evaluation.heldout import Heldout
        with self.assertRaises(ValueError):Heldout('missing','missing','missing',partition='test')
    @unittest.skipUnless(importlib.util.find_spec('transformers') and Path('outputs/action_sft/model/tokenizer.json').is_file(), 'local Qwen tokenizer not available')
    def test_real_qwen_chat_template_mask(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('outputs/action_sft/model',local_files_only=True)
        target=first_target([]);payload={'request':'hello','context':{},'history':[],'tools':[],'observations':[]}
        result=render(tokenizer,payload,target,1536);n=result['prompt_tokens']
        self.assertEqual(result['labels'][:n],[-100]*n);self.assertIn(tokenizer.eos_token_id,result['labels'][n:])
        self.assertEqual(json.loads(tokenizer.decode(result['input_ids'][n:],skip_special_tokens=True).strip()),target)
    def test_response_only_mask_and_eos(self):
        tok=ToyTokenizer();target=first_target([]);payload={'request':'hello','context':{},'history':[],'tools':[],'observations':[]}
        item=render(tok,payload,target,5000);n=item['prompt_tokens']
        self.assertEqual(item['labels'][:n],[-100]*n);self.assertEqual(item['labels'][n:],item['input_ids'][n:]);self.assertIn(tok.eos_token_id,item['labels'][n:]);self.assertEqual(item['labels'][-1],10)
    def test_target_never_truncated(self):
        tok=ToyTokenizer();target=first_target([]);payload={'request':'x'*500,'context':{},'history':[],'tools':[],'observations':[]}
        with self.assertRaisesRegex(ValueError,'too_long'):render(tok,payload,target,100)
    def test_routed_compaction_is_label_blind_and_keeps_system(self):
        class CounterTokenizer:
            def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
                payload=json.loads(messages[1]['content']);return [1]*(100+100*len(payload['history'])+150*len(payload['tools']))
        tools=[{'name':'sys.ask_missing_info','parameters':{'required':['tool_name','params']}}]+[{'name':'a.'+str(i),'parameters':{'required':['id']}} for i in range(8)]
        payload={'request':'test','context':{},'history':[{'role':'user','text':'old'}]*2,'tools':tools,'observations':[]}
        result,size,count=compact_routed(payload,CounterTokenizer(),limit=1200,reserve=256)
        self.assertGreater(count,0);self.assertLessEqual(size+256,1200);self.assertEqual(result['history'],[])
        self.assertEqual(len(result['tools']),4);self.assertEqual(result['tools'][0]['parameters']['required'],['tool_name','params'])
    def test_inference_adapter_is_required(self):
        class M:pass
        with self.assertRaises(ValueError):TransformersActionModel(M(),None)
    def test_paired_metrics(self):
        labels={'a':{'tool_calls':[{'name':'x','arguments':{'y':1}}]},'b':{'tool_calls':[]}}
        rows=[{'id':'a','proposal':{'action':'call','name':'x','arguments':{'y':1}},'latency_seconds':1,'candidates':['x']},{'id':'b','proposal':{'action':'finish','name':'','arguments':{}},'latency_seconds':2,'candidates':[]}]
        result=summarize(rows,labels);self.assertEqual(result['first_action_exact'],1);self.assertEqual(result['first_tool_accuracy'],1);self.assertEqual(result['finish_accuracy'],1)
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch not installed')
    def test_resume_checkpoint_restores_rng_and_checks_identity(self):
        import torch
        from src.toolcall.training.checkpoint import save_resume,load_resume
        param=torch.nn.Parameter(torch.tensor([1.0]));optimizer=torch.optim.AdamW([param]);rng=random.Random(17)
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available',return_value=False):
            path=Path(tmp)/'resume.pt';save_resume(path,optimizer,rng,{'step':2,'order':[1,0]},'identity-a')
            expected=rng.random();rng.random();state=load_resume(path,optimizer,rng,'identity-a')
            self.assertEqual(state['step'],2);self.assertEqual(rng.random(),expected)
            with self.assertRaisesRegex(ValueError,'identity'):load_resume(path,optimizer,rng,'different')
    def test_model_artifacts_ignored_by_policy(self):
        gitignore=Path('.gitignore').read_text(encoding='utf-8');self.assertIn('/outputs/*',gitignore);self.assertIn('*.safetensors',gitignore)

if __name__=='__main__':unittest.main()
