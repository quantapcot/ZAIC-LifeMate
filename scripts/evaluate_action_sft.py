"""Compare a base checkpoint and its active adapter on the same local inputs."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from scripts.evaluate_action_agent import ROOT,WORKTREE
from src.toolcall.agent.action_agent import ActionAgent,RESPONSE,SYSTEM,available_tools,compact_tool,safe_history
from src.toolcall.agent.tool_router import ToolRouter
from src.toolcall.agent.toolenv_adapter import open_train_env
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.heldout import Heldout
from src.toolcall.evaluation.metrics import metrics
from src.toolcall.training.transformers_backend import TransformersActionModel,load_paired


def target_of(label):
    calls=label['tool_calls'];return {'action':'call','name':calls[0]['name'],'arguments':calls[0]['arguments']} if calls else {'action':'finish','name':'','arguments':{}}


def summarize(rows,labels):
    total=len(rows);valid=sum(r.get('json_valid',r.get('proposal') is not None) for r in rows);schema_valid=sum(r.get('proposal') is not None for r in rows);action=tool=args=exact=0;times=[];first_calls=0;retrieval_misses=0;finish_n=finish_ok=missing_n=missing_ok=0
    for r in rows:
        target=target_of(labels[r['id']]);proposal=r.get('proposal');times.append(r['latency_seconds'])
        if target['action']=='call':first_calls+=1;retrieval_misses+=target['name'] not in r.get('candidates',[])
        if target['action']=='finish':finish_n+=1;finish_ok+=proposal is not None and proposal.get('action')=='finish'
        if target['name']=='sys.ask_missing_info':missing_n+=1;missing_ok+=proposal is not None and proposal.get('name')=='sys.ask_missing_info'
        if proposal is None:continue
        action+=proposal['action']==target['action'];tool+=proposal['action']=='call' and target['action']=='call' and proposal['name']==target['name'];args+=proposal['action']=='call' and target['action']=='call' and proposal['name']==target['name'] and proposal['arguments']==target['arguments'];exact+=proposal==target
    div=lambda x,y:x/y if y else None
    return {'samples':total,'valid_json_rate':div(valid,total),'schema_valid_rate':div(schema_valid,total),'correct_action_type':div(action,total),'first_tool_accuracy':div(tool,first_calls),'first_tool_denominator':first_calls,'first_argument_exact':div(args,first_calls),'first_action_exact':div(exact,total),'retrieval_miss':retrieval_misses,'finish_accuracy':div(finish_ok,finish_n),'missing_info_name_accuracy':div(missing_ok,missing_n),'mean_latency_seconds':div(sum(times),len(times)),'prompt_budget_failures':sum(r.get('error')=='PromptBudgetError' for r in rows),'mean_prompt_tokens':div(sum(r.get('prompt_tokens',0) for r in rows),sum('prompt_tokens' in r for r in rows)),'errors':{e:sum(r.get('error')==e for r in rows) for e in sorted({r.get('error') for r in rows if r.get('error')})}}


def routed_payload(row,reg):
    context=row['context'];ctx={k:context[k] for k in ('current_time','timezone','location','user_profile') if k in context}
    names,_=ToolRouter(reg,8).rank(available_tools(context,reg),row['audio_text'],ctx)
    return {'request':row['audio_text'],'context':ctx,'history':safe_history(row['history']),'tools':[compact_tool(reg.get_tool(n)) for n in names],'observations':[]}


def compact_routed(payload,tokenizer,limit=2048,reserve=256):
    # Match the Agent's label-blind budget policy: trim history first, then shortlist.
    compacted=0
    while True:
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        size=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        if size+reserve<=limit:return payload,size,compacted
        if payload['history']:
            payload['history']=payload['history'][len(payload['history'])//2+1:];compacted+=1;continue
        if len(payload['tools'])>4:
            payload['tools']=[x for x in payload['tools'] if x['name']=='sys.ask_missing_info']+[x for x in payload['tools'] if x['name']!='sys.ask_missing_info'][:3];compacted+=1;continue
        return payload,size,compacted


def attempt(backend,fn):
    t=time.perf_counter();backend.last_text=''
    try:proposal,usage=fn();return {'proposal':proposal,'json_valid':True,'latency_seconds':time.perf_counter()-t,'output_hash':hashlib.sha256(backend.last_text.encode()).hexdigest(),'output_tokens':usage.get('output_tokens')}
    except Exception as exc:
        valid_json=False
        if backend.last_text:
            try:json.loads(backend.last_text,parse_constant=lambda value: (_ for _ in ()).throw(ValueError('invalid JSON constant')));valid_json=True
            except ValueError:pass
        return {'proposal':None,'json_valid':valid_json,'latency_seconds':time.perf_counter()-t,'error':getattr(exc,'code',type(exc).__name__),'output_hash':hashlib.sha256(backend.last_text.encode()).hexdigest() if backend.last_text else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train-run',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--output-dir',required=True);parser.add_argument('--selection',default='outputs/action_sft/b1-evaluation-selection.json');parser.add_argument('--samples',type=int,default=8);parser.add_argument('--agent-samples',type=int,default=3);a=parser.parse_args()
    out=(WORKTREE/a.output_dir).resolve();base=(WORKTREE/'outputs/action_sft').resolve()
    if base not in out.parents:raise ValueError('output must be under outputs/action_sft')
    train=json.loads(((WORKTREE/a.train_run)/'train_manifest.json').read_text(encoding='utf-8'));dataset=(WORKTREE/a.dataset);manifest=json.loads((dataset/'dataset_manifest.json').read_text(encoding='utf-8'))
    rows=[json.loads(x) for x in (dataset/'dev.jsonl').read_text(encoding='utf-8').splitlines()]
    # The first eight dev records were used for checkpoint selection; reserve different dev records for this report.
    chosen=rows[8:8+a.samples]
    if len(chosen)!=a.samples or a.agent_samples>a.samples:raise ValueError('insufficient held-out TRAIN-dev examples')
    dev_ids=[r['id'] for r in chosen]
    selection=json.loads((WORKTREE/a.selection).read_text(encoding='utf-8'))
    if dev_ids!=selection['dev_ids']:raise ValueError('TRAIN-dev IDs differ from predeclared selection')
    train_data=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train')
    val=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='val')
    prior=WORKTREE.parent/'lifemate-action-agent/outputs/agent/c1-pilot-final/run_manifest.json'
    if not prior.is_file():raise ValueError('prior validation manifest required to reserve fresh IDs')
    old=set(json.loads(prior.read_text(encoding='utf-8'))['selected_ids']);excluded=old|set(selection.get('prior_val_ids',[]));val_ids=selection['val_ids']
    if len(val_ids)!=a.samples or len(set(val_ids))!=len(val_ids) or not set(val_ids)<=(val.ids-excluded) or not set(dev_ids)<=train_data.ids:raise ValueError('evaluation IDs do not match predeclared selection')
    out.mkdir(parents=True,exist_ok=False)
    (out/'evaluation_manifest.json').write_text(json.dumps({'dev_ids':dev_ids,'fresh_val_ids':val_ids,'prior_pilot_excluded':True,'train_selected_adapter':train['selected_adapter'],'models':['base_adapter_disabled','adapter_active'],'seed':42,'temperature':0,'max_new_tokens':256,'context_limit':2048,'dataset_sha256':manifest['dataset_sha256']},indent=2),encoding='utf-8')
    model,tok=load_paired(train['model_dir'],train['selected_adapter']);backend=TransformersActionModel(model,tok);reg=ToolRegistry.from_catalog(ROOT/'train/train_tools_catalog.json')
    import torch
    adapter_norm=sum(float(p.detach().abs().sum().cpu()) for name,p in model.named_parameters() if 'lora_B' in name)
    if adapter_norm<=0:raise ValueError('adapter appears untrained')
    # Compare next-token logits on the same real prompt, even when greedy text happens to match.
    probe=torch.tensor([chosen[0]['input_ids'][:chosen[0]['prompt_tokens']]],device='cuda',dtype=torch.long)
    with torch.inference_mode():
        with model.disable_adapter():base_logits=model(input_ids=probe,attention_mask=torch.ones_like(probe),use_cache=False,logits_to_keep=1).logits.float().cpu()
        tuned_logits=model(input_ids=probe,attention_mask=torch.ones_like(probe),use_cache=False,logits_to_keep=1).logits.float().cpu()
    logit_delta=float((base_logits-tuned_logits).abs().max())
    if logit_delta==0:raise RuntimeError('adapter produced identical probe logits')
    del probe,base_logits,tuned_logits
    outputs={'controlled':{'base':[],'adapter':[]},'routed_dev':{'base':[],'adapter':[]},'routed_val':{'base':[],'adapter':[]},'agent':{'base':[],'adapter':[]}};failures=[]
    for row in chosen:
        idx=row['prompt_tokens']; prompt_ids=row['input_ids'][:idx]
        # Candidate membership for controlled scoring is implicit by construction; record placeholder names after labels are read.
        for mode in ('base','adapter'):
            backend.use_adapter=mode=='adapter'; result=attempt(backend,lambda:backend.generate_ids(prompt_ids,RESPONSE));result.update(id=row['id'],candidates=[]);outputs['controlled'][mode].append(result)
        print('controlled',len(outputs['controlled']['base']),'/',len(chosen),flush=True)
    for group,data,ids in [('routed_dev',train_data,dev_ids),('routed_val',val,val_ids)]:
        for key in ids:
            payload,prompt_tokens,compactions=compact_routed(routed_payload(data.records[key],reg),tok);candidates=[x['name'] for x in payload['tools']]
            for mode in ('base','adapter'):
                backend.use_adapter=mode=='adapter';result=attempt(backend,lambda:backend.generate_json(SYSTEM,json.dumps(payload,ensure_ascii=False),RESPONSE));result.update(id=key,candidates=candidates,prompt_tokens=prompt_tokens,compactions=compactions);outputs[group][mode].append(result)
            print(group,len(outputs[group]['base']),'/',len(ids),flush=True)
    for key in dev_ids[:a.agent_samples]:
        row=train_data.records[key]
        for mode in ('base','adapter'):
            backend.use_adapter=mode=='adapter'
            try:
                env=open_train_env(ROOT,key);result=ActionAgent(backend,reg,deadline_seconds=30,max_steps=4).run(row['audio_text'],row['context'],row['history'],env);result['id']=key;outputs['agent'][mode].append(result)
            except Exception as exc:failures.append({'mode':mode,'id':key,'error':type(exc).__name__})
        print('agent',len(outputs['agent']['base']),'/',a.agent_samples,flush=True)
    # Only after predictions are complete are held-out labels loaded for scoring.
    train_data.finish_predictions();val.finish_predictions();dev_labels=train_data.labels(dev_ids);val_labels=val.labels(val_ids)
    report={'base_model':train['base_model'],'selected_adapter':train['selected_adapter'],'adapter_lora_b_abs_sum':adapter_norm,'probe_max_logit_difference':logit_delta,'controlled_policy':'gold-positive TRAIN-dev candidates; not production','old_validation_excluded':True,'groups':{},'agent_failures':len(failures)}
    for group in ('controlled','routed_dev','routed_val'):
        labs=val_labels if group=='routed_val' else dev_labels
        report['groups'][group]={}
        for mode in ('base','adapter'):
            rr=outputs[group][mode]
            if group=='controlled':
                for r in rr:
                    target=target_of(labs[r['id']]);r['candidates']=[target['name']] if target['name'] else []
            report['groups'][group][mode]=summarize(rr,labs)
        b=outputs[group]['base'];f=outputs[group]['adapter'];report['groups'][group]['differing_raw_output_hashes']=sum(x['output_hash'] is not None and y['output_hash'] is not None and x['output_hash']!=y['output_hash'] for x,y in zip(b,f))
    report['groups']['agent']={m:metrics(outputs['agent'][m],dev_labels) for m in ('base','adapter')}
    (out/'predictions.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8');(out/'errors.json').write_text(json.dumps(failures,indent=2),encoding='utf-8');(out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
