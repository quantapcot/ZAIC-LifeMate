"""Predeclare label-blind TRAIN-dev and fresh validation IDs for paired SFT evaluation."""
import argparse
import json
import random
from scripts.evaluate_action_agent import ROOT,WORKTREE
from src.toolcall.evaluation.heldout import Heldout


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output-file',required=True);p.add_argument('--exclude-selection',action='append',default=[]);p.add_argument('--samples',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--checkpoint-dev-prefix',type=int,default=8);a=p.parse_args()
    out=(WORKTREE/a.output_file).resolve();base=(WORKTREE/'outputs/action_sft').resolve()
    if base not in out.parents or a.samples<1 or a.checkpoint_dev_prefix<0:raise ValueError('invalid output/count')
    rows=[json.loads(x) for x in ((WORKTREE/a.dataset)/'dev.jsonl').read_text(encoding='utf-8').splitlines()];dev=[r['id'] for r in rows[a.checkpoint_dev_prefix:a.checkpoint_dev_prefix+a.samples]]
    data=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='val')
    pilot=WORKTREE.parent/'lifemate-action-agent/outputs/agent/c1-pilot-final/run_manifest.json'
    if not pilot.is_file():raise ValueError('prior validation manifest required')
    old=set(json.loads(pilot.read_text(encoding='utf-8'))['selected_ids']);prior=set()
    for path in a.exclude_selection:
        previous=json.loads((WORKTREE/path).read_text(encoding='utf-8'));prior.update(previous['val_ids']);prior.update(previous.get('prior_val_ids',[]))
    candidates=sorted(data.ids-old-prior);random.Random(a.seed).shuffle(candidates);val=candidates[:a.samples]
    if len(dev)!=a.samples or len(val)!=a.samples:raise ValueError('insufficient eligible records')
    result={'dev_ids':dev,'val_ids':val,'prior_val_ids':sorted(prior),'seed':a.seed,'checkpoint_selection_dev_prefix_count':a.checkpoint_dev_prefix}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'dev_count':len(dev),'fresh_val_count':len(val),'excluded_pilot_count':len(old),'excluded_previous_count':len(prior)}))
if __name__=='__main__':main()
