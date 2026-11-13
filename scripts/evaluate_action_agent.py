"""Evaluate the text-first agent on the train-derived validation partition."""
import argparse
import json
import random
from pathlib import Path
from src.toolcall.agent.action_agent import ActionAgent, available_tools
from src.toolcall.agent.toolenv_adapter import open_train_env
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.heldout import Heldout
from src.toolcall.evaluation.metrics import metrics
from src.toolcall.llm.ollama_client import OllamaClient, MODEL

WORKTREE = Path(__file__).resolve().parents[1]
ROOT = WORKTREE.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--sdk-smoke', action='store_true')
    parser.add_argument('--split', choices=['val'], default='val'); parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--output-dir', default='outputs/agent/c1-validation')
    parser.add_argument('--model', default=MODEL); parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--context-length', type=int, default=16384); parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()
    data = Heldout(ROOT/'train/train.jsonl', ROOT/'train/train_tools_catalog.json', WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits')
    registry = ToolRegistry.from_catalog(ROOT/'train/train_tools_catalog.json')
    if args.dry_run:
        print(json.dumps({'validation_count':len(data.ids),'catalog_tools':len(registry.list_tools()),'hashes_verified':True})); return
    if args.sdk_smoke:
        row=data.records[sorted(data.ids)[0]]; env=open_train_env(ROOT,row['id'])
        names=available_tools(row['context'],registry)
        target=next(n for n in names if n != 'sys.ask_missing_info' and registry.get_schema(n)['properties'])
        field=next(iter(registry.get_schema(target)['properties']))
        verdict=env.validate('sys.ask_missing_info',{'tool_name':target,'params':[field]})
        if verdict.get('ok') is not True: raise RuntimeError('SDK validation failed: '+str(verdict))
        result=env.call('sys.ask_missing_info',{'tool_name':target,'params':[field]})
        calls=env.get_tool_calls(); state_hash=env.get_state_hash()
        if len(calls)!=1 or not isinstance(state_hash,str) or not state_hash: raise RuntimeError('SDK result invalid')
        print(json.dumps({'sdk_validation':True,'sdk_call_returned':result is not None,'recorded_calls':len(calls),'state_hash_returned':True})); return
    if args.limit is not None and args.limit < 1: parser.error('--limit must be positive')
    base=(WORKTREE/'outputs/agent').resolve(); out=(WORKTREE/args.output_dir).resolve()
    if out == base or base not in out.parents: parser.error('output directory must be under outputs/agent')
    out.mkdir(parents=True,exist_ok=False)
    ids=sorted(data.ids); random.Random(args.seed).shuffle(ids); ids=ids[:args.limit] if args.limit else ids
    model=OllamaClient(model=args.model,seed=args.seed,temperature=0,context_length=args.context_length,max_tokens=args.max_tokens,timeout=args.timeout)
    agent=ActionAgent(model,registry,deadline_seconds=args.timeout)
    predictions=[]; failures=[]
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for index,key in enumerate(ids):
            row=data.records[key]
            try:
                env=open_train_env(ROOT,key)
                result=agent.run(row['audio_text'],row['context'],row['history'],env)
                result['id']=key; predictions.append(result)
                f.write(json.dumps({'id':key,'tool_calls':result['tool_calls'],'state_hash':result['state_hash']},ensure_ascii=False)+'\n'); f.flush()
            except Exception as exc:
                failures.append({'id':key,'error_type':type(exc).__name__})
            print(f'Processed {index+1}/{len(ids)}',flush=True)
    data.finish_predictions(); labels=data.labels([p['id'] for p in predictions])
    report=metrics(predictions,labels); report['failed_samples']=len(failures); report['requested_samples']=len(ids)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'errors.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    (out/'runtime.json').write_text(json.dumps([{k:r[k] for k in ('id','error','latency_seconds','model_seconds','tool_seconds','output_tokens','invalid_proposals','error_code','sdk_checks','sdk_valid','executions','execution_success')} for r in predictions],indent=2),encoding='utf-8')
    (out/'run_manifest.json').write_text(json.dumps({'model':args.model,'model_digest':model.model_digest(),'seed':args.seed,'temperature':0,'context_length':args.context_length,'max_tokens':args.max_tokens,'deadline_seconds':args.timeout,'split':'val','dataset_sha256':data.manifest['dataset_sha256'],'catalog_sha256':data.manifest['catalog_sha256'],'selected_ids':ids},indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__ == '__main__': main()
