"""Run a bounded, local-only ToolLoop pilot; outputs are provisional."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import os
import time
from src.toolcall.data import ToolRegistry, TrainOnlyDataset
from src.toolcall.llm import OllamaClient, TransportError, InvalidModelOutput
from src.toolcall.synthesis.pipeline import generate
from src.toolcall.synthesis import sampler
from src.toolcall.synthesis.schema_inspection import catalog_eligibility
from src.toolcall.verification import SemanticVerifier, POLICY_VERSION
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
           'rejected':sum(r['status']=='rejected' for r in records), 'verified_candidate':sum(r['status']=='verified_candidate' for r in records), 'manual_review_required':sum(r['status']=='manual_review_required' for r in records), 'scenarios':dict(Counter(r['scenario_type'] for r in records)),
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
    histories=[step for r in records for h in r.get('stage_history',{}).values() for step in h]
    if histories:
        generator_usage=[step.get('generation_usage') or {} for step in histories]
        verifier_usage=[(step.get('semantic') or {}).get('usage') or {} for step in histories]
        def total(values, key):
            nums=[v[key] for v in values if isinstance(v.get(key),(int,float))]
            return sum(nums) if nums else None
        stats['generator_tokens']=total(generator_usage,'output_tokens')
        stats['verifier_tokens']=total(verifier_usage,'output_tokens')
        stats['generator_latency_seconds']=total(generator_usage,'latency_seconds')
        stats['verifier_latency_seconds']=total(verifier_usage,'latency_seconds')
        stats['deterministic_failure_steps']=sum(step.get('deterministic') is not None and not step['deterministic']['valid'] for step in histories)
        stats['semantic_failure_steps']=sum(step.get('semantic') is not None and not step['semantic'].get('passed') for step in histories)
        stats['uncertain_records']=sum(r['status']=='manual_review_required' for r in records)
        stats['refinements_by_record']=[sum(max(0,len(h)-1) for h in r.get('stage_history',{}).values()) for r in records]
        stats['initial_pass_records']=sum(r['status']=='verified_candidate' and all(len(h)==1 for h in r['stage_history'].values()) for r in records)
        stats['recovered_records']=sum(r['status']=='verified_candidate' and any(len(h)>1 for h in r['stage_history'].values()) for r in records)
        stats['failure_codes']=dict(Counter(issue.get('error_code','UNKNOWN') for step in histories for issue in step.get('issues',[])))
        accepted=[r for r in records if r['status']=='verified_candidate']
        initial_failures=[r for r in records if any(h and (not (h[0].get('deterministic') or {}).get('valid',False) or not (h[0].get('semantic') or {}).get('passed',False)) for h in r.get('stage_history',{}).values())]
        stats['verified_acceptance_rate']=len(accepted)/len(records) if records else None
        stats['initial_pass_rate']=stats['initial_pass_records']/len(records) if records else None
        stats['refinement_recovery_rate']=stats['recovered_records']/len(initial_failures) if initial_failures else None
        stats['mean_refinements_per_accepted']=statistics.mean(sum(max(0,len(h)-1) for h in r['stage_history'].values()) for r in accepted) if accepted else None
        stats['maximum_refinements_reached']=max((len(h)-1 for r in records for h in r.get('stage_history',{}).values()),default=0)
        stats['deterministic_failure_rate']=stats['deterministic_failure_steps']/len(histories) if histories else None
        semantic_attempts=sum(step.get('semantic') is not None for step in histories)
        stats['semantic_failure_rate']=stats['semantic_failure_steps']/semantic_attempts if semantic_attempts else None
        stats['verifier_uncertainty_rate']=stats['uncertain_records']/len(records) if records else None
    stats['note']='Engineering metrics only; records are not SFT-ready.'
    return stats


def run_outcome(records, selection):
    if not records or any(r['failure_reason']=='TransportError' or any(issue.get('source') in ('transport','verifier_runtime') for history in r.get('stage_history',{}).values() for step in history for issue in step.get('issues',[])) for r in records):
        return 'failed'
    return 'partial' if selection['shortfall'] else 'completed'


def write_run(out, records, stats):
    out=output_path(out)
    if stats.get('outcome') not in ('completed','partial','failed'):
        raise ValueError('explicit run outcome required')
    out.mkdir(parents=True,exist_ok=False)
    # A missing completion marker makes an interrupted run visibly incomplete.
    files=[('records.json',records),('summary.json',stats)]
    if stats.get('policy_version'):
        worksheet=[{'record_index':i,'status':r['status'],'repaired':any(len(h)>1 for h in r.get('stage_history',{}).values()),'human_review':'not_reviewed'} for i,r in enumerate(records)]
        files.append(('review.json',worksheet))
    for name,obj in files:
        with (out/name).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
    marker={'completed':'COMPLETE','partial':'PARTIAL','failed':'FAILED'}[stats['outcome']]
    temporary=out/'.outcome.tmp'
    with temporary.open('x',encoding='utf-8') as f: f.write('run outcome only; records are not SFT-ready\n')
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
    parser.add_argument('--verify',action='store_true',help='enable separate development-only semantic verifier')
    parser.add_argument('--verifier-model',default='qwen3:8b-q4_K_M')
    parser.add_argument('--max-refinements',type=int,default=3)
    args=parser.parse_args()
    if not 1 <= args.limit <= 10: parser.error('limit must be between 1 and 10')
    if not 0 <= args.max_refinements <= 3: parser.error('max-refinements must be between 0 and 3')
    if args.verify and args.model == args.verifier_model: parser.error('verifier and generator models must differ')
    try:
        client=OllamaClient(model=args.model,base_url=args.base_url,seed=args.seed,context_length=args.context_length,max_tokens=args.max_tokens,timeout=args.timeout,keep_alive=0 if args.verify else None)
        verifier_client=OllamaClient(model=args.verifier_model,base_url=args.base_url,seed=args.seed,temperature=0,context_length=args.context_length,max_tokens=args.max_tokens,timeout=args.timeout,keep_alive=0,think=False) if args.verify else None
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
            print(json.dumps({'train_count':len(train),'catalog_tools':len(registry.list_tools()),'eligible':eligible,'catalog_schema_filter':catalog_eligibility(registry),'unsupported':['parallel','parallel_multiple'],'model':client.model,'verifier_model':args.verifier_model if args.verify else None,'max_refinements':args.max_refinements}));return
        if args.verify:
            gen_digest=client.model_digest()
            verifier_digest=verifier_client.model_digest()
            if not gen_digest or not verifier_digest: raise ValueError('generator or verifier model unavailable locally')
            if gen_digest==verifier_digest: raise ValueError('generator and verifier have the same model digest')
            if args.smoke:
                client.unload()
                verifier_client.unload()
        if args.smoke:
            raw,usage=client.generate_json('Return JSON only.','Return the JSON object {"ok": true}.',{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']})
            if raw != {'ok':True}: raise InvalidModelOutput('smoke output failed validation')
            result={'smoke':'passed','latency_seconds':usage['latency_seconds'],'output_tokens':usage['output_tokens']}
            if args.verify:
                vraw,vusage=verifier_client.generate_json('Return JSON only.','Return the JSON object {\"ok\": true}.',{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']})
                if vraw != {'ok':True}: raise InvalidModelOutput('verifier smoke failed')
                result['verifier_smoke']='passed';result['verifier_latency_seconds']=vusage['latency_seconds']
            print(json.dumps(result));return
        output_path(args.output_dir)
        start=time.perf_counter()
        records,selection=generate(train,registry,client,args.limit,args.seed,with_report=True,verifier=SemanticVerifier(verifier_client) if args.verify else None,max_refinements=args.max_refinements)
        wall_seconds=time.perf_counter()-start
        stats=summarize(records,client,args.seed,args.limit)
        stats['total_wall_seconds']=wall_seconds
        stats.update(selection)
        stats['outcome']=run_outcome(records,selection)
        stats['model_digest']=client.model_digest()
        if args.verify:
            stats['verifier_model']=args.verifier_model;stats['verifier_digest']=verifier_digest;stats['policy_version']=POLICY_VERSION;stats['max_refinements']=args.max_refinements
            for record in records:
                record['model_metadata']={'generator':args.model,'generator_digest':gen_digest,'generator_options':client.options,'verifier':args.verifier_model,'verifier_digest':verifier_digest,'verifier_options':verifier_client.options,'generator_keep_alive':client.keep_alive,'verifier_keep_alive':verifier_client.keep_alive,'verifier_think':verifier_client.think}
        stats['catalog_version']=registry.version
        stats['dataset_sha256']=json.loads((args.splits_dir/'split_manifest.json').read_text(encoding='utf-8'))['dataset_sha256']
        write_run(args.output_dir,records,stats)
        print(json.dumps(stats,ensure_ascii=True,indent=2))
        if stats['outcome'] != 'completed': raise RuntimeError('pilot outcome: '+stats['outcome'])
    except (ValueError,FileNotFoundError,FileExistsError,TransportError,InvalidModelOutput,RuntimeError) as exc:
        print(f'Error: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)


if __name__=='__main__': main()
