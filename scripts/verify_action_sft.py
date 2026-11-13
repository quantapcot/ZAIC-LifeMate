"""Verify a prepared TRAIN-only dataset against gold, tokenizer and loss-mask invariants."""
import argparse
import collections
import json
from scripts.evaluate_action_agent import ROOT,WORKTREE
from src.toolcall.evaluation.heldout import Heldout
from src.toolcall.training.dataset_builder import first_target,group_key


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--model-dir',default='outputs/action_sft/model');a=p.parse_args()
    directory=(WORKTREE/a.dataset).resolve();base=(WORKTREE/'outputs/action_sft').resolve()
    if base not in directory.parents:raise ValueError('dataset must be under outputs/action_sft')
    manifest=json.loads((directory/'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('source_partition')!='train':raise ValueError('TRAIN only')
    rows={q:[json.loads(x) for x in (directory/(q+'.jsonl')).read_text(encoding='utf-8').splitlines()] for q in ('fit','dev')}
    d=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train');ids=[r['id'] for group in rows.values() for r in group]
    if len(ids)!=len(set(ids)) or not set(ids)<=d.ids or manifest['dataset_sha256']!=d.manifest['dataset_sha256'] or manifest['catalog_sha256']!=d.manifest['catalog_sha256']:raise ValueError('identity or split violation')
    d.finish_predictions();labels=d.labels(ids)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str((WORKTREE/a.model_dir).resolve()),local_files_only=True);counts=collections.Counter();groups={}
    for part,records in rows.items():
        for r in records:
            n=r['prompt_tokens'];full=r['input_ids'];decoded=tok.decode(full[n:],skip_special_tokens=True).strip();target=json.loads(decoded)
            text=tok.decode(full[:n],skip_special_tokens=False);raw=text.split('<|im_start|>user\n',1)[1].rsplit('<|im_end|>\n<|im_start|>assistant\n',1)[0];payload=json.loads(raw)
            calls=labels[r['id']]['tool_calls'];names=[t['name'] for t in payload['tools']]
            checks={'gold_match':target==first_target(calls),'mask_match':r['labels']==[-100]*n+full[n:],'eos_present':tok.eos_token_id in full[n:],'within_limit':len(full)<=manifest['max_length'],'prompt_fields_valid':set(payload)=={'request','context','history','tools','observations'},'empty_observations':payload['observations']==[],'gold_candidate_present':not calls or calls[0]['name'] in names}
            for k,v in checks.items():counts[k]+=bool(v)
            counts['history_trimmed']+=len(payload['history'])<min(12,len(d.records[r['id']]['history']))
            group=group_key(d.records[r['id']]['audio_text']);previous=groups.setdefault(group,part);counts['cross_partition_groups']+=previous!=part
    total=len(ids)
    for k in ('gold_match','mask_match','eos_present','within_limit','prompt_fields_valid','empty_observations','gold_candidate_present'):
        if counts[k]!=total:raise ValueError('dataset verification failed: '+k)
    if counts['cross_partition_groups']:raise ValueError('duplicate group crosses fit/dev')
    report={'examples':total,'counts':dict(counts),'attention_mask':'un-padded single examples use all ones','dataset_sha256':d.manifest['dataset_sha256'],'catalog_sha256':d.manifest['catalog_sha256']}
    path=directory/'verification.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))
if __name__=='__main__':main()
