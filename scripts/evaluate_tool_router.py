"""TRAIN-only, model-free routing diagnostics; output goes to ignored local files."""
import argparse
import json
import time
from pathlib import Path
from scripts.evaluate_action_agent import ROOT,WORKTREE
from src.toolcall.agent.action_agent import available_tools,compact_tool
from src.toolcall.agent.tool_router import ToolRouter
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.heldout import Heldout
from src.toolcall.evaluation.routing_diagnostics import recall_at_k


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ids-from-manifest',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    data=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train')
    ids=json.loads(Path(a.ids_from_manifest).read_text(encoding='utf-8'))['selected_ids']
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=data.ids: raise ValueError('TRAIN IDs required')
    out=(WORKTREE/a.output_dir).resolve();base=(WORKTREE/'outputs/agent').resolve()
    if base not in out.parents: raise ValueError('output must be under outputs/agent')
    out.mkdir(parents=True,exist_ok=False)
    reg=ToolRegistry.from_catalog(ROOT/'train/train_tools_catalog.json');ranked={};times=[]
    # Complete all independent rankings before loading any labels.
    for key in ids:
        row=data.records[key]; names=available_tools(row['context'],reg)
        ctx={k:row['context'][k] for k in ('current_time','timezone','location','user_profile') if k in row['context']}
        t=time.perf_counter();order,scores=ToolRouter(reg,top_k=max(1,len(names))).rank(names,row['audio_text'],ctx);times.append(time.perf_counter()-t)
        ranked[key]={'order':order,'scores':scores,'installed_count':len(names)}
    data.finish_predictions();labels=data.labels(ids)
    report={'samples':len(ids),'router_mean_seconds':sum(times)/len(times),'categories_without_ground_truth':'dependency, language of description, Vietnamese paraphrase'}; detail=[]
    for k in (4,8,12,16):
        hits=total=first_hits=first_total=all_hits=nonempty=prompt_chars=0;groups={'single':[0,0],'multi':[0,0]}
        for key in ids:
            gold=[c['name'] for c in labels[key]['tool_calls']];r=ranked[key];order=[n for n in r['order'] if n!='sys.ask_missing_info'];choice=order[:k]+(['sys.ask_missing_info'] if 'sys.ask_missing_info' in r['order'] else [])
            h,n=recall_at_k(choice,gold,len(choice));hits+=h;total+=n
            if gold:
                nonempty+=1;first_total+=1;first_hits+=gold[0] in choice;all_hits+=h==n
                g='single' if n==1 else 'multi';groups[g][0]+=h;groups[g][1]+=n
            prompt_chars+=len(json.dumps([compact_tool(reg.get_tool(n)) for n in choice],ensure_ascii=False))
        report[str(k)]={'recall':hits/total if total else None,'gold_calls':total,'hits':hits,'first_recall':first_hits/first_total if first_total else None,'first_hits':first_hits,'first_total':first_total,'all_gold_coverage':all_hits/nonempty if nonempty else None,'avg_schema_codepoints':prompt_chars/len(ids),'single_recall':groups['single'][0]/groups['single'][1] if groups['single'][1] else None,'multi_recall':groups['multi'][0]/groups['multi'][1] if groups['multi'][1] else None}
    for key in ids:
        g=[c['name'] for c in labels[key]['tool_calls']];r=ranked[key];detail.append({'id':key,'gold_names':g,'installed_count':r['installed_count'],'ranked':r['order'],'scores':r['scores'],'gold_ranks':{n:r['order'].index(n)+1 if n in r['order'] else None for n in set(g)}})
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(out/'details.json').write_text(json.dumps(detail,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
