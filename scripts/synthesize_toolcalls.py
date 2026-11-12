"""Run a bounded, local-only ToolLoop pilot; outputs are provisional."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import os
from src.toolcall.data import ToolRegistry, TrainOnlyDataset
from src.toolcall.llm import OllamaClient, TransportError, InvalidModelOutput
from src.toolcall.synthesis.pipeline import generate
from src.toolcall.synthesis import sampler
from src.toolcall.synthesis.schema_inspection import catalog_eligibility
from scripts.prepare_toolloop_data import DEFAULT_TRAIN, OUTPUT

ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS_ROOT = OUTPUT / 'synthesis'


def output_path(path):
    base=SYNTHESIS_ROOT
    if base.resolve()!=base: raise ValueError('unsafe synthesis directory')
    output=Path(path).resolve()
    if output==base or not output.is_relative_to(base): raise ValueError('output must be a new directory inside outputs/toolloop/synthesis')
    if output.exists(): raise FileExistsError('output directory already exists')
    return output


def summarize(records, client, seed, requested):
    stages=('ground_truth','backward_query','forward_calls')
    stats={'attempted':len(records),'requested':requested,'provisional':sum(r['status']=='provisional' for r in records),
           'rejected':sum(r['status']=='rejected' for r in records), 'scenarios':dict(Counter(r['scenario_type'] for r in records)),
           'failure_reasons':dict(Counter(str(r['failure_stage'])+':'+r['failure_reason'] for r in records if r['failure_reason'])),
           'seed':seed,'model':client.model,'options':client.options,'base_url':client.base_url,'timeout_seconds':client.timeout,'peak_gpu_memory':'unavailable', 'stages':{}}
    for stage in stages:
        values=[r['stages'][stage] for r in records if stage in r['stages']]
        latencies=[v['latency_seconds'] for v in values]
        tokens=[v.get('usage',{}).get('output_tokens') for v in values]
        stats['stages'][stage]={'completed':len(values),'valid':sum(v['validation']['valid'] for v in values),
             'mean_latency_seconds':statistics.mean(latencies) if latencies else None,
             'output_tokens':sum(t for t in tokens if isinstance(t,int)) if any(isinstance(t,int) for t in tokens) else None}
    stats['forward_json_valid']=len([r for r in records if 'forward_calls' in r['stages']])
    stats['note']='Engineering metrics only; provisional outputs are not verified training data.'
    return stats


def run_outcome(records, selection):
    if not records or any(r['failure_reason']=='TransportError' for r in records):
        return 'failed'
    return 'partial' if selection['shortfall'] else 'completed'


def write_run(out, records, stats):
    out=output_path(out)
    if stats.get('outcome') not in ('completed','partial','failed'):
        raise ValueError('explicit run outcome required')
    out.mkdir(parents=True,exist_ok=False)
    # A missing completion marker makes an interrupted run visibly incomplete.
    for name,obj in [('records.json',records),('summary.json',stats)]:
        with (out/name).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
    marker={'completed':'COMPLETE','partial':'PARTIAL','failed':'FAILED'}[stats['outcome']]
    temporary=out/'.outcome.tmp'
    with temporary.open('x',encoding='utf-8') as f: f.write('run outcome only; records are provisional or rejected\n')
    os.replace(temporary, out/marker)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for name in ('dry-run','smoke','generate'): modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--train-dir',type=Path,default=DEFAULT_TRAIN)
    parser.add_argument('--splits-dir',type=Path,default=OUTPUT/'splits')
    parser.add_argument('--output-dir',type=Path,default=SYNTHESIS_ROOT/'pilot-seed42')
    parser.add_argument('--model',default='qwen3:4b-instruct-2507-q4_K_M')
    parser.add_argument('--base-url',default='http://127.0.0.1:11434')
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--context-length',type=int,default=4096)
    parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if not 1 <= args.limit <= 10: parser.error('limit must be between 1 and 10')
    try:
        client=OllamaClient(model=args.model,base_url=args.base_url,seed=args.seed,context_length=args.context_length,max_tokens=args.max_tokens,timeout=args.timeout)
        registry=ToolRegistry.from_catalog(args.train_dir/'train_tools_catalog.json')
        train=TrainOnlyDataset.from_prepared_split(args.train_dir/'train.jsonl',args.train_dir/'train_tools_catalog.json',args.splits_dir)
        if args.dry_run:
            eligible = {'simple': 0, 'multiple': 0}
            for record in train.iter_records():
                for scenario in eligible:
                    try:
                        sampler.sample(record, registry, scenario, args.seed)
                        eligible[scenario] += 1
                    except ValueError:
                        continue
            if not all(eligible.values()): raise ValueError('no eligible examples for a supported scenario')
            print(json.dumps({'train_count':len(train),'catalog_tools':len(registry.list_tools()),'eligible':eligible,'catalog_schema_filter':catalog_eligibility(registry),'unsupported':['parallel','parallel_multiple'],'model':client.model}));return
        if args.smoke:
            raw,usage=client.generate_json('Return JSON only.','Return the JSON object {"ok": true}.',{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']})
            if raw != {'ok':True}: raise InvalidModelOutput('smoke output failed validation')
            print(json.dumps({'smoke':'passed','latency_seconds':usage['latency_seconds'],'output_tokens':usage['output_tokens']}));return
        output_path(args.output_dir)
        records,selection=generate(train,registry,client,args.limit,args.seed,with_report=True)
        stats=summarize(records,client,args.seed,args.limit)
        stats.update(selection)
        stats['outcome']=run_outcome(records,selection)
        stats['model_digest']=client.model_digest()
        stats['catalog_version']=registry.version
        stats['dataset_sha256']=json.loads((args.splits_dir/'split_manifest.json').read_text(encoding='utf-8'))['dataset_sha256']
        write_run(args.output_dir,records,stats)
        print(json.dumps(stats,ensure_ascii=True,indent=2))
        if stats['outcome'] != 'completed': raise RuntimeError('pilot outcome: '+stats['outcome'])
    except (ValueError,FileNotFoundError,FileExistsError,TransportError,InvalidModelOutput,RuntimeError) as exc:
        print(f'Error: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)


if __name__=='__main__': main()
