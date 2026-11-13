"""DIAGNOSTIC_ORACLE_TRAIN_ONLY: first-action comparison; never used by the runtime agent."""
import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from scripts.evaluate_action_agent import ROOT,WORKTREE
from src.toolcall.agent.action_agent import SYSTEM,RESPONSE,available_tools,compact_tool,safe_history
from src.toolcall.agent.tool_router import ToolRouter,tokens
from src.toolcall.agent.toolenv_adapter import open_train_env
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.heldout import Heldout
from src.toolcall.llm.ollama_client import OllamaClient,MODEL


def legacy_rank(reg,available,request,ctx,k=8):
    query=Counter(tokens(request+' '+json.dumps(ctx,ensure_ascii=False)+' []'));meta={}
    for n in available:
        if n=='sys.ask_missing_info':continue
        tool=reg.get_tool(n);meta[n]=Counter(tokens(n.replace('_',' ')*2+' '+tool.get('description','')+' '+json.dumps(tool['parameters'],ensure_ascii=False)))
    df=Counter(t for words in meta.values() for t in words);total=max(1,len(meta))
    scores={n:sum((1+math.log(1+c))*min(query[t],2)*(1+math.log((total+1)/(df[t]+1))) for t,c in words.items() if t in query) for n,words in meta.items()}
    chosen=sorted(scores,key=lambda n:(-scores[n],n))[:k]
    return chosen+(['sys.ask_missing_info'] if 'sys.ask_missing_info' in available else [])


def oracle_candidates(partition, available, ordinary, target, seed):
    if partition!='train': raise ValueError('DIAGNOSTIC_ORACLE_TRAIN_ONLY')
    if not isinstance(target,str) or target not in available: raise ValueError('oracle target must be an installed name')
    app=[x for x in ordinary if x!='sys.ask_missing_info']
    if target!='sys.ask_missing_info' and target not in app:
        if app: app[-1]=target
        else: app=[target]
    random.Random(seed).shuffle(app)
    if 'sys.ask_missing_info' in available: app.append('sys.ask_missing_info')
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ids-from-manifest',required=True);parser.add_argument('--output-dir',required=True);a=parser.parse_args()
    # Deliberately no split option: access is hard-coded and verified as TRAIN.
    data=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train')
    ids=json.loads(Path(a.ids_from_manifest).read_text(encoding='utf-8'))['selected_ids']
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=data.ids: raise ValueError('TRAIN partition IDs required')
    out=(WORKTREE/a.output_dir).resolve();base=(WORKTREE/'outputs/agent').resolve()
    if base not in out.parents: raise ValueError('output must be under outputs/agent')
    out.mkdir(parents=True,exist_ok=False)
    data.finish_predictions();labels=data.labels(ids)
    reg=ToolRegistry.from_catalog(ROOT/'train/train_tools_catalog.json');model=OllamaClient(model=MODEL,seed=42,temperature=0,context_length=16384,max_tokens=256,timeout=30)
    # Only the target tool name is used for oracle candidate construction; no gold arguments or sequence enter prompts.
    rows=[]; eligible=[key for key in ids if labels[key]['tool_calls']]
    for index,key in enumerate(eligible):
        row=data.records[key];gold=labels[key]['tool_calls'][0];target=gold['name'];avail=available_tools(row['context'],reg)
        if target not in avail: rows.append({'id':key,'skipped':'gold_not_installed'});continue
        ctx={k:row['context'][k] for k in ('current_time','timezone','location','user_profile') if k in row['context']}
        improved,_=ToolRouter(reg,8).rank(avail,row['audio_text'],ctx)
        modes={'A_c1_1':legacy_rank(reg,avail,row['audio_text'],ctx),'B_c1_2':improved,'C_DIAGNOSTIC_ORACLE_TRAIN_ONLY':oracle_candidates('train',avail,improved,target,42+index)}
        for mode,candidates in modes.items():
            payload={'request':row['audio_text'],'context':ctx,'history':safe_history(row['history']),'tools':[compact_tool(reg.get_tool(n)) for n in candidates],'observations':[]}
            start=time.perf_counter();item={'id':key,'mode':mode,'candidate_count':len(candidates),'target_present':target in candidates}
            try:
                env=open_train_env(ROOT,key)
                proposal,usage=model.generate_json(SYSTEM,json.dumps(payload,ensure_ascii=False),RESPONSE)
                item.update(action=proposal.get('action'),proposed_tool=proposal.get('name'),arguments_exact=proposal.get('action')=='call' and proposal.get('name')==target and proposal.get('arguments')==gold['arguments'],correct_tool=proposal.get('action')=='call' and proposal.get('name')==target,prompt_estimate=(usage.get('input_budget') or {}).get('estimated_input_tokens'))
                if proposal.get('action')=='call' and proposal.get('name') in candidates and isinstance(proposal.get('arguments'),dict): item['sdk_valid']=env.validate(proposal['name'],proposal['arguments']).get('ok') is True
            except Exception as exc: item['error']=getattr(exc,'code',type(exc).__name__);item['prompt_estimate']=(getattr(model,'last_budget',None) or {}).get('estimated_input_tokens')
            item['latency_seconds']=time.perf_counter()-start;rows.append(item)
        print(f'Processed {index+1}/{len(eligible)}',flush=True)
    report={'label':'DIAGNOSTIC_ORACLE_TRAIN_ONLY','eligible_samples':len(eligible),'model':MODEL,'model_digest':model.model_digest(),'modes':{}}
    for mode in ('A_c1_1','B_c1_2','C_DIAGNOSTIC_ORACLE_TRAIN_ONLY'):
        rr=[r for r in rows if r.get('mode')==mode]; n=len(rr);present=[r for r in rr if r['target_present']];correct=sum(r.get('correct_tool',False) for r in rr)
        report['modes'][mode]={'samples':n,'candidate_present':len(present),'first_tool_correct':correct,'first_tool_accuracy':correct/n if n else None,'accuracy_given_present':sum(r.get('correct_tool',False) for r in present)/len(present) if present else None,'correct_arguments':sum(r.get('arguments_exact',False) for r in rr),'finish':sum(r.get('action')=='finish' for r in rr),'wrong_tool':sum(r.get('action')=='call' and not r.get('correct_tool') for r in rr),'errors':dict(Counter(r.get('error') for r in rr if r.get('error'))),'sdk_valid':sum(r.get('sdk_valid') is True for r in rr),'sdk_checked':sum('sdk_valid' in r for r in rr),'mean_latency':sum(r['latency_seconds'] for r in rr)/n if n else None,'mean_prompt_estimate':sum(r['prompt_estimate'] for r in rr if r.get('prompt_estimate') is not None)/max(1,sum(r.get('prompt_estimate') is not None for r in rr))}
    (out/'diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
