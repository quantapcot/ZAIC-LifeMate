"""Paired A/B/C research harness. All artifacts stay local and unreviewed."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from src.toolcall.data import ToolRegistry, TrainOnlyDataset
from src.toolcall.data.split import query_key
from src.toolcall.llm import OllamaClient
from src.toolcall.synthesis import pipeline
from src.toolcall.verification import SemanticVerifier, POLICY_VERSION
from scripts.prepare_toolloop_data import DEFAULT_TRAIN, OUTPUT

ROOT=OUTPUT/'ablation'


def path_for_output(path):
    if ROOT.resolve()!=ROOT: raise ValueError('unsafe ablation root')
    out=Path(path).resolve()
    if out==ROOT or not out.is_relative_to(ROOT): raise ValueError('output must be under outputs/toolloop/ablation')
    if out.exists(): raise FileExistsError('ablation output already exists')
    return out


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def technical_error(result):
    if result.get('failure_reason') in ('TransportError','PROMPT_TOO_LARGE'): return True
    return any(i.get('error_code') in ('OUT_OF_MEMORY','TIMEOUT','TRANSPORT_FAILURE','UNLOAD_UNCONFIRMED','PROMPT_TOO_LARGE')
               for history in result.get('stage_history',{}).values() for step in history for i in step.get('issues',[]))


def paired(train, registry, generator, verifier, limit=3, seed=42, max_refinements=3):
    """Select once, freeze A, replay that exact initial candidate for B and C."""
    if not 1 <= limit <= 5 or not 0 <= max_refinements <= 3: raise ValueError('invalid experiment bounds')
    vclient=verifier.client
    if generator.model==vclient.model: raise ValueError('distinct models required')
    ghash,vhash=generator.model_digest(),vclient.model_digest()
    if not ghash or not vhash or ghash==vhash: raise ValueError('distinct local model digests required')
    generator.keep_alive=0;vclient.keep_alive=0
    generator.confirm_unloaded=True;vclient.confirm_unloaded=True
    generator.unload();vclient.unload()
    meta={'generator':generator.model,'generator_digest':ghash,'generator_options':copy.deepcopy(generator.options),
          'verifier':vclient.model,'verifier_digest':vhash,'verifier_options':copy.deepcopy(vclient.options),
          'generator_keep_alive':0,'verifier_keep_alive':0,'verifier_think':vclient.think}
    records,selected,eligible,excluded,fallbacks=pipeline.select_scenarios(train,registry,limit,seed)
    queries={query_key(r['audio_text']) for r in records}
    manifest={'experiment_id':str(uuid.uuid4()),'seed':seed,'policy_version':POLICY_VERSION,'models':meta,
              'scenarios':[],'eligible':eligible,'exclusions':dict(excluded),'fallbacks':dict(fallbacks)}
    initial=[];results_b=[];results_c=[];stop=None;malformed_verdicts=0;start=time.perf_counter()
    invalid_generator_outputs=0
    for index,(source,scenario) in enumerate(selected):
        sample_seed=seed+index
        ground=pipeline.sampler.sample(source,registry,scenario,sample_seed)
        from dataclasses import asdict
        manifest['scenarios'].append({'id':f'scenario-{index:03d}','source_train_id':source['id'],'scenario_type':scenario,
                                     'seed':sample_seed,'ground_truth':asdict(ground),'ground_hash':digest_json(asdict(ground))})
        sample=pipeline.run_attempt(source,registry,generator,scenario,sample_seed,queries)
        sample['scenario_id']=manifest['scenarios'][-1]['id']
        initial.append(sample)
        if technical_error(sample):stop='technical_failure_in_A';break
        if sample.get('failure_reason')=='InvalidModelOutput': invalid_generator_outputs+=1
        if invalid_generator_outputs>=2:stop='repeated_invalid_generator_output';break
    for index,sample in enumerate(initial):
        if stop:break
        source,scenario=selected[index]; sample_seed=seed+index
        # No invented downstream initial output when mode A terminated early.
        if not all(name in sample['stages'] for name in ('ground_truth','backward_query','forward_calls')):
            censored={'status':'not_evaluable','reason':'missing_frozen_initial_stage','scenario_id':manifest['scenarios'][index]['id']}
            results_b.append(copy.deepcopy(censored));results_c.append(copy.deepcopy(censored));continue
        for output,budget in ((results_b,0),(results_c,max_refinements)):
            result=pipeline.run_verified_attempt(source,registry,generator,scenario,sample_seed,queries,verifier,budget,meta,initial_candidate=copy.deepcopy(sample))
            result['scenario_id']=manifest['scenarios'][index]['id']
            result['initial_candidate_sha256']=digest_json(sample)
            output.append(result)
            if technical_error(result):stop='technical_failure_in_verification';break
            malformed_verdicts += sum(issue.get('source')=='verifier_runtime' and issue.get('error_code') in ('INVALID_JSON','OUTPUT_SCHEMA_MISMATCH','MALFORMED_ENVELOPE','INVALID_VERDICT') for history in result.get('stage_history',{}).values() for step in history for issue in step.get('issues',[]))
            if malformed_verdicts >= 2:stop='repeated_invalid_verifier_output';break
    for i,sample in enumerate(initial):
        manifest['scenarios'][i]['initial_candidate_sha256']=digest_json(sample)
    def counts(items, mode):
        from collections import Counter
        count=len(items)
        evaluated=[x for x in items if x['status']!='not_evaluable']
        histories=[step for item in items for history in item.get('stage_history',{}).values() for step in history]
        stage_entries=[stage for item in items for stage in item.get('stages',{}).values()]
        usages=([stage.get('usage') or {} for stage in stage_entries] if mode=='A' else [step.get('generation_usage') or {} for step in histories])
        semantic=[step.get('semantic') for step in histories if step.get('semantic') is not None]
        verifier_usages=[x.get('usage') or {} for x in semantic]
        def total(values,key):
            nums=[v[key] for v in values if isinstance(v.get(key),(int,float))]
            return sum(nums) if nums else None
        def rate(n,d): return n/d if d else None
        valid_json=sum((isinstance(item.get('stages',{}).get('forward_calls',{}).get('model_output'),dict) if mode=='A' else any(isinstance(step.get('output'),dict) and isinstance(step['output'].get('tool_calls'),list) for step in item.get('stage_history',{}).get('forward_calls',[]))) for item in items)
        valid_schema=sum(bool(item.get('stages',{}).get('forward_calls',{}).get('validation',{}).get('valid')) for item in items)
        accepted=sum(item['status']=='verified_candidate' for item in items)
        refinements=[sum(max(0,len(history)-1) for history in item.get('stage_history',{}).values()) for item in items]
        failed_initial=sum(any(h and ((h[0].get('deterministic') is not None and not h[0]['deterministic']['valid']) or (h[0].get('semantic') is not None and not h[0]['semantic'].get('passed'))) for h in item.get('stage_history',{}).values()) for item in items)
        recovered=sum(item['status']=='verified_candidate' and any(len(h)>1 for h in item.get('stage_history',{}).values()) for item in items)
        failures=Counter(issue.get('error_code','UNKNOWN') for item in items for history in item.get('stage_history',{}).values() for step in history for issue in step.get('issues',[]))
        failures.update(item['failure_reason'] for item in items if item.get('failure_reason') and not item.get('stage_history'))
        return {'scenarios':count,'initial_generation_attempts':count if mode=='A' else 0,'status_counts':dict(Counter(x['status'] for x in items)),
                'structured_json_valid_rate':rate(valid_json,count),'deterministic_schema_valid_rate':rate(valid_schema,count),
                'semantic_acceptance_rate':rate(accepted,len(evaluated)) if mode!='A' else None,
                'initial_candidates_reused':len(evaluated) if mode!='A' else 0,
                'initial_pass_rate':rate(sum(x['status']=='verified_candidate' and all(len(h)==1 for h in x.get('stage_history',{}).values()) for x in evaluated),len(evaluated)) if mode!='A' else None,
                'rejected':sum(x['status']=='rejected' for x in items),'uncertain':sum(x['status']=='manual_review_required' for x in items),
                'failure_codes':dict(failures),'refinements':sum(refinements),'refinement_recovery_rate':rate(recovered,failed_initial) if mode!='A' else None,
                'maximum_refinements_reached':max((len(h)-1 for item in items for h in item.get('stage_history',{}).values()),default=0),
                'generator_input_tokens':total(usages,'prompt_tokens'),'generator_output_tokens':total(usages,'output_tokens'),
                'verifier_input_tokens':total(verifier_usages,'prompt_tokens'),'verifier_output_tokens':total(verifier_usages,'output_tokens'),
                'generator_latency_seconds':total(usages,'latency_seconds'),'verifier_latency_seconds':total(verifier_usages,'latency_seconds'),
                'model_load_unload_overhead_seconds':None,'peak_gpu_memory':None}
    summary={'mode_A':counts(initial,'A'),'mode_B':counts(results_b,'B'),'mode_C':counts(results_c,'C'),
             'paired_initial_count':len(initial),'wall_seconds':time.perf_counter()-start,'stop_reason':stop,
             'outcome':'partial' if stop or len(initial)<limit else 'completed',
             'human_review':'not_reviewed','independent_quality':'unavailable','sft_ready':False}
    worksheet=[{'scenario_id':s['id'],'status':'not_reviewed','rubric':['tool correctness','intent consistency','argument grounding','completeness','extra actions','Vietnamese naturalness','unsupported assumptions','false acceptance','false rejection']} for s in manifest['scenarios']]
    return {'experiment_manifest.json':manifest,'initial_candidates.json':initial,'results_A.json':copy.deepcopy(initial),
            'results_B.json':results_b,'results_C.json':results_c,'summary.json':summary,'human_review.json':worksheet,
            'limitations.json':{'paired_initial':'B and C replay frozen A; downstream is regenerated only after upstream changes',
                                'not_evaluable':'Missing initial outputs are not invented or cherry-picked',
                                'quality':'Verifier outcomes are treatment, not independent human correctness',
                                'model_runtime':'model sampling may be nondeterministic','human_review':'not_reviewed'}}


def write_artifacts(output, artifacts):
    out=path_for_output(output);out.mkdir(parents=True,exist_ok=False)
    for name, obj in artifacts.items():
        with (out/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    marker='COMPLETE' if artifacts['summary.json']['outcome']=='completed' else 'PARTIAL'
    tmp=out/'.outcome.tmp'
    with tmp.open('x',encoding='utf-8') as f:f.write('artifact write finished; inspect summary outcome; no SFT approval\n')
    os.replace(tmp,out/marker)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=3);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-refinements',type=int,default=3)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'phase3-1-pilot')
    parser.add_argument('--train-dir',type=Path,default=DEFAULT_TRAIN);parser.add_argument('--splits-dir',type=Path,default=OUTPUT/'splits')
    parser.add_argument('--verifier-model',default='qwen3:8b-q4_K_M')
    parser.add_argument('--context-length',type=int,default=4096)
    parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--max-payload-bytes',type=int,default=262144)
    parser.add_argument('--token-estimate-factor',type=float,default=1.0)
    args=parser.parse_args()
    path_for_output(args.output_dir)
    reg=ToolRegistry.from_catalog(args.train_dir/'train_tools_catalog.json')
    train=TrainOnlyDataset.from_prepared_split(args.train_dir/'train.jsonl',args.train_dir/'train_tools_catalog.json',args.splits_dir)
    options={'seed':args.seed,'context_length':args.context_length,'max_tokens':args.max_tokens,'max_payload_bytes':args.max_payload_bytes,'token_estimate_factor':args.token_estimate_factor}
    generator=OllamaClient(keep_alive=0,**options);verifier=SemanticVerifier(OllamaClient(model=args.verifier_model,temperature=0,keep_alive=0,think=False,**options))
    artifacts=paired(train,reg,generator,verifier,args.limit,args.seed,args.max_refinements)
    prepared=json.loads((args.splits_dir/'split_manifest.json').read_text(encoding='utf8'))
    artifacts['experiment_manifest.json']['dataset_sha256']=prepared['dataset_sha256']
    artifacts['experiment_manifest.json']['catalog_sha256']=prepared['catalog_sha256']
    write_artifacts(args.output_dir,artifacts)
    print(json.dumps(artifacts['summary.json'],indent=2))
    if artifacts['summary.json']['outcome']!='completed': raise SystemExit(1)


if __name__=='__main__':main()
