"""Prepare a TRAIN-only, gold-positive first-action Qwen dataset in ignored storage."""
import argparse
import collections
import json
from pathlib import Path
from scripts.evaluate_action_agent import ROOT,WORKTREE
from src.toolcall.agent.action_agent import basic_issues,available_tools
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.heldout import Heldout
from src.toolcall.training.dataset_builder import first_target,partition,prepare_prompt,render


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--model-dir',default='outputs/action_sft/model');p.add_argument('--max-length',type=int,default=1536);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    out=(WORKTREE/a.output_dir).resolve();base=(WORKTREE/'outputs/action_sft').resolve()
    if base not in out.parents:raise ValueError('output must be under outputs/action_sft')
    data=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train')
    reg=ToolRegistry.from_catalog(ROOT/'train/train_tools_catalog.json')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str((WORKTREE/a.model_dir).resolve()),local_files_only=True,trust_remote_code=False)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:raise RuntimeError('Qwen chat template or EOS missing')
    out.mkdir(parents=True,exist_ok=False);data.finish_predictions();labels=data.labels(data.ids)
    counts=collections.Counter(); lengths=[]; fit=[];dev=[]; ids={'fit':[],'dev':[]}
    for key in sorted(data.ids):
        row=data.records[key];label=labels[key];calls=label.get('tool_calls');results=label.get('tool_results')
        if not isinstance(calls,list) or not isinstance(results,list):counts['malformed_label']+=1;continue
        counts['zero' if not calls else 'single' if len(calls)==1 else 'multi']+=1
        counts['aligned_result_count' if len(calls)==len(results) else 'misaligned_result_count']+=1
        counts['missing_info_record']+=any(isinstance(c,dict) and c.get('name')=='sys.ask_missing_info' for c in calls)
        counts['zero_with_history']+=not calls and bool(row.get('history'))
        if not isinstance(row.get('audio_text'),str) or not row['audio_text'].strip():counts['skip_empty_transcript']+=1;continue
        try:
            available=available_tools(row['context'],reg)
            if any(not isinstance(c,dict) or not isinstance(c.get('name'),str) or not isinstance(c.get('arguments'),dict) for c in calls):raise ValueError('malformed_call')
            if any(c['name'] not in reg.list_tools() for c in calls):raise ValueError('unknown_tool')
            if any(c['name'] not in available for c in calls):raise ValueError('unavailable_tool')
            target=first_target(calls)
            if calls and basic_issues(target['arguments'],reg.get_schema(target['name'])):raise ValueError('first_call_structural_failure')
            prompt=prepare_prompt(row,reg,target,seed=a.seed)
            encoded=render(tokenizer,prompt,target,a.max_length)
        except (ValueError,KeyError,TypeError) as exc:
            counts['skip_'+str(exc)]+=1;continue
        part=partition(row['audio_text'],a.seed);encoded['id']=key;(fit if part=='fit' else dev).append(encoded);ids[part].append(key);lengths.append(len(encoded['input_ids']));counts['usable_'+part]+=1
    for name,rows in [('fit',fit),('dev',dev)]:
        with (out/(name+'.jsonl')).open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    manifest={'dataset_sha256':data.manifest['dataset_sha256'],'catalog_sha256':data.manifest['catalog_sha256'],'source_partition':'train','seed':a.seed,'fit_dev_method':'sha256 of seed and NFKC casefold whitespace-normalized transcript, 15% dev','model':'Qwen/Qwen3-4B-Instruct-2507','tokenizer_local':a.model_dir,'max_length':a.max_length,'gold_positive_candidates':True,'zero_call_policy':'finish supervision; no assumption that empty gold implies out-of-scope','counts':dict(counts),'max_tokens':max(lengths,default=0),'selected_ids':ids}
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in manifest.items() if k!='selected_ids'}))
if __name__=='__main__':main()
