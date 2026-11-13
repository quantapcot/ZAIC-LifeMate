"""TRAIN-dev-only, deliberately stratified finish and missing-info diagnostics."""
import argparse
import json
import random
from scripts.evaluate_action_agent import ROOT,WORKTREE
from scripts.evaluate_action_sft import attempt,summarize
from src.toolcall.agent.action_agent import RESPONSE,ActionAgent
from src.toolcall.agent.toolenv_adapter import open_train_env
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.metrics import metrics
from src.toolcall.evaluation.heldout import Heldout
from src.toolcall.training.transformers_backend import TransformersActionModel,load_paired


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train-run',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--per-category',type=int,default=4);p.add_argument('--val-selection');p.add_argument('--val-agent-samples',type=int,default=3);a=p.parse_args()
    out=(WORKTREE/a.output_dir).resolve();root=(WORKTREE/'outputs/action_sft').resolve()
    if root not in out.parents or a.per_category<1:raise ValueError('invalid output or count')
    d=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train')
    rows=[json.loads(line) for line in ((WORKTREE/a.dataset)/'dev.jsonl').read_text(encoding='utf-8').splitlines()][16:]
    if not {r['id'] for r in rows}<=d.ids:raise ValueError('non-TRAIN ID')
    # Label-based stratification is explicitly diagnostic and never used in training or production.
    d.finish_predictions();labels=d.labels([r['id'] for r in rows]);groups={'zero':[],'missing_info':[]}
    for row in rows:
        calls=labels[row['id']]['tool_calls']
        if not calls:groups['zero'].append(row)
        elif calls[0]['name']=='sys.ask_missing_info':groups['missing_info'].append(row)
    for name in groups:
        random.Random(42).shuffle(groups[name]);groups[name]=groups[name][:a.per_category]
    if any(not x for x in groups.values()):raise ValueError('requested diagnostic category unavailable')
    out.mkdir(parents=True,exist_ok=False);(out/'selection.json').write_text(json.dumps({k:[r['id'] for r in v] for k,v in groups.items()},indent=2),encoding='utf-8')
    train=json.loads(((WORKTREE/a.train_run)/'train_manifest.json').read_text(encoding='utf-8'));model,tok=load_paired(train['model_dir'],train['selected_adapter']);backend=TransformersActionModel(model,tok);report={'policy':'TRAIN-dev, label-stratified and gold-positive; not representative','groups':{}}
    all_outputs={}
    for group,examples in groups.items():
        all_outputs[group]={'base':[],'adapter':[]}
        for row in examples:
            for mode in ('base','adapter'):
                backend.use_adapter=mode=='adapter';result=attempt(backend,lambda:backend.generate_ids(row['input_ids'][:row['prompt_tokens']],RESPONSE));result.update(id=row['id'],candidates=[labels[row['id']]['tool_calls'][0]['name']] if labels[row['id']]['tool_calls'] else []);all_outputs[group][mode].append(result)
            print(group,len(all_outputs[group]['base']),'/',len(examples),flush=True)
        report['groups'][group]={mode:summarize(all_outputs[group][mode],labels) for mode in ('base','adapter')}
    if a.val_selection:
        selection=json.loads((WORKTREE/a.val_selection).read_text(encoding='utf-8'));ids=selection['val_ids'][:a.val_agent_samples]
        val=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='val')
        if not ids or len(ids)!=a.val_agent_samples or not set(ids)<=val.ids:raise ValueError('invalid validation selection')
        registry=ToolRegistry.from_catalog(ROOT/'train/train_tools_catalog.json');autonomous={'base':[],'adapter':[]}
        for key in ids:
            row=val.records[key]
            for mode in ('base','adapter'):
                backend.use_adapter=mode=='adapter';env=open_train_env(ROOT,key);result=ActionAgent(backend,registry,deadline_seconds=30,max_steps=4).run(row['audio_text'],row['context'],row['history'],env);result['id']=key;autonomous[mode].append(result)
            print('val_agent',len(autonomous['base']),'/',len(ids),flush=True)
        val.finish_predictions();val_labels=val.labels(ids);report['fresh_validation_agent']={mode:metrics(autonomous[mode],val_labels) for mode in ('base','adapter')};all_outputs['fresh_validation_agent']=autonomous
    (out/'predictions.json').write_text(json.dumps(all_outputs,indent=2),encoding='utf-8');(out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
