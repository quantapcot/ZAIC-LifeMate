"""Gold-supervised first-action data construction; never used by runtime routing."""
import hashlib
import json
import random
import unicodedata
from src.toolcall.agent.action_agent import SYSTEM,available_tools,basic_issues,compact_tool,safe_history
from src.toolcall.agent.tool_router import ToolRouter


def group_key(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def partition(text,seed=42):
    h=hashlib.sha256((str(seed)+'\0'+group_key(text)).encode()).digest()
    return 'dev' if int.from_bytes(h[:4],'big')%100<15 else 'fit'


def first_target(calls):
    if not calls:return {'action':'finish','name':'','arguments':{}}
    call=calls[0]
    return {'action':'call','name':call['name'],'arguments':call['arguments']}


def prepare_prompt(row,registry,target,top_k=8,seed=42):
    context=row['context'];available=available_tools(context,registry)
    if target['action']=='call' and target['name'] not in available: raise ValueError('gold_tool_not_installed')
    ctx={k:context[k] for k in ('current_time','timezone','location','user_profile') if k in context}
    names,_=ToolRouter(registry,top_k).rank(available,row['audio_text'],ctx)
    app=[n for n in names if n!='sys.ask_missing_info']
    target_name=target['name']
    if target_name and target_name!='sys.ask_missing_info' and target_name not in app:
        if app:app[-1]=target_name
        else:app=[target_name]
    # Use a deterministic permutation so gold is not consistently in a marked position.
    rng=random.Random(int.from_bytes(hashlib.sha256((str(seed)+row['id']).encode()).digest()[:8],'big'));rng.shuffle(app)
    names=app+(['sys.ask_missing_info'] if 'sys.ask_missing_info' in available else [])
    return {'request':row['audio_text'],'context':ctx,'history':safe_history(row['history']),'tools':[compact_tool(registry.get_tool(n)) for n in names],'observations':[]}


def render(tokenizer,payload,target,max_length):
    target_json=json.dumps(target,ensure_ascii=False,separators=(',',':'))
    assert json.loads(target_json)==target
    working=json.loads(json.dumps(payload,ensure_ascii=False))
    while True:
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(working,ensure_ascii=False)}]
        prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':target_json}],tokenize=True,add_generation_prompt=False)
        if full[:len(prompt)]!=prompt: raise ValueError('chat_template_prefix_mismatch')
        if tokenizer.eos_token_id not in full[len(prompt):]: raise ValueError('chat_template_eos_mismatch')
        if len(full)<=max_length:
            return {'input_ids':full,'labels':[-100]*len(prompt)+full[len(prompt):],'prompt_tokens':len(prompt),'target_tokens':len(full)-len(prompt),'candidate_count':len(working['tools'])}
        if working['history']:
            working['history']=working['history'][1:];continue
        tools=working['tools']; removable=[i for i,t in enumerate(tools) if t['name'] not in (target['name'],'sys.ask_missing_info')]
        if len(removable)>1:tools.pop(removable[-1]);continue
        raise ValueError('prompt_or_target_too_long')
