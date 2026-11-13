"""Three bounded stages; validation is preliminary and has no side effects."""
import random
import time
from dataclasses import asdict
from collections import Counter
from . import sampler, query_generator, call_generator, validators
from .errors import issue
from src.toolcall.llm import TransportError, InvalidModelOutput
from src.toolcall.data.split import query_key
from src.toolcall.data import TrainOnlyDataset
from src.toolcall.data.contracts import GroundTruth, Metadata
from src.toolcall.verification import POLICY_VERSION
from src.toolcall.verification.semantic_verifier import inspect_verdict
from .refinement import run_stage


def run_attempt(record, registry, client, scenario, seed, training_queries=frozenset()):
    result={'status':'rejected','scenario_type':scenario,'seed':seed,'stages':{},'failure_stage':None,'failure_reason':None}
    start=time.perf_counter()
    try:
        ground=sampler.sample(record,registry,scenario,seed)
        check=validators.validate_ground(ground,registry)
        result['stages']['ground_truth']={'output':asdict(ground),'validation':asdict(check),'latency_seconds':time.perf_counter()-start}
        if not check.valid: raise ValueError('invalid_ground_truth')
    except ValueError as exc:
        result['failure_stage']=1;result['failure_reason']=str(exc)
        path=exc.issues[0]['path'] if isinstance(exc,sampler.UnsupportedSchema) else 'ground_truth'
        if 'ground_truth' in result['stages']:
            result['feedback']=result['stages']['ground_truth']['validation']
        else:
            result['feedback']=asdict(validators.feedback(1,[issue(str(exc),path,source='sampler',retryable=False)],('semantic_coherence','independence')))
        if isinstance(exc,sampler.UnsupportedSchema): result['unsupported_schema']={'tool_name':exc.tool_name,'constraints':exc.issues}
        return result
    try:
        raw,query,usage=query_generator.generate(client,ground,registry)
        check=validators.validate_query(raw,ground)
        if check.valid and query_key(query.user_query) in training_queries:
            check=validators.feedback(2, [('user_query','copies_training_query')], ('semantic_alignment', 'argument_grounding'))
        result['stages']['backward_query']={'output':asdict(query),'model_output':raw,'validation':asdict(check),'usage':usage,'latency_seconds':usage['latency_seconds']}
        if not check.valid: raise ValueError('invalid_query')
    except (TransportError,InvalidModelOutput,ValueError) as exc:
        result['failure_stage']=2;result['failure_reason']=type(exc).__name__ if isinstance(exc,(TransportError,InvalidModelOutput)) else 'invalid_query'
        if isinstance(exc,(TransportError,InvalidModelOutput)):
            item=issue(exc.code,exc.field,source='transport' if isinstance(exc,TransportError) else 'model_output',retryable=isinstance(exc,InvalidModelOutput))
            result['feedback']=asdict(validators.feedback(2,[item],('semantic_alignment','argument_grounding')))
            if exc.raw_output is not None: result['local_invalid_output_excerpt']=exc.raw_output
        else: result['feedback']=result['stages'].get('backward_query',{}).get('validation',asdict(validators.feedback(2,[issue('OUTPUT_SCHEMA_MISMATCH','response',source='model_output')])))
        return result
    try:
        raw,calls,usage=call_generator.generate(client,query.user_query,ground.candidate_tools,registry,ground.metadata.dataset_version)
        check=validators.validate_calls(raw,ground,registry)
        result['stages']['forward_calls']={'output':asdict(calls) if calls else None,'model_output':raw,'validation':asdict(check),'usage':usage,'latency_seconds':usage['latency_seconds']}
        if not check.valid: raise ValueError('invalid_calls')
    except (TransportError,InvalidModelOutput,ValueError) as exc:
        result['failure_stage']=3;result['failure_reason']=type(exc).__name__ if isinstance(exc,(TransportError,InvalidModelOutput)) else 'invalid_calls'
        if isinstance(exc,(TransportError,InvalidModelOutput)):
            item=issue(exc.code,exc.field,source='transport' if isinstance(exc,TransportError) else 'model_output',retryable=isinstance(exc,InvalidModelOutput))
            result['feedback']=asdict(validators.feedback(3,[item],('semantic_alignment','argument_grounding')))
            if exc.raw_output is not None: result['local_invalid_output_excerpt']=exc.raw_output
        else: result['feedback']=result['stages'].get('forward_calls',{}).get('validation',asdict(validators.feedback(3,[issue('OUTPUT_SCHEMA_MISMATCH','response',source='model_output')])))
        return result
    result['status']='provisional'
    return result



def run_verified_attempt(record, registry, client, scenario, seed, training_queries, verifier, max_refinements=3, model_metadata=None):
    if verifier is None: raise ValueError('semantic verifier required')
    required=('generator','generator_digest','verifier','verifier_digest')
    if not isinstance(model_metadata,dict) or not all(isinstance(model_metadata.get(k),str) and model_metadata[k] for k in required) or model_metadata['generator_digest']==model_metadata['verifier_digest']:
        raise ValueError('distinct generator and verifier identities and digests required')
    result={'status':'rejected','sft_ready':False,'policy_version':POLICY_VERSION,'scenario_type':scenario,
            'seed':seed,'stages':{},'stage_history':{},'failure_stage':None,'failure_reason':None,'model_metadata':model_metadata}
    def store(stage, name, phase):
        result['stage_history'][name]=phase['history']
        if phase['history']:
            last=phase['history'][-1]
            result['stages'][name]={'output':phase['output'], 'validation':last.get('deterministic') or {'valid':False},
                'latency_seconds':sum(x.get('latency_seconds',0) for x in phase['history']), 'usage':last.get('generation_usage') or {}}
        if phase['status']!='passed':
            result['failure_stage']=stage
            result['failure_reason']='verification_'+phase['status']
            result['status']=phase['status'] if phase['status']=='manual_review_required' else 'rejected'
            return False
        return True
    def verify_stage(stage, selected_ground, query='', calls=None):
        decision=verifier.verify(stage,selected_ground,registry,query=query,calls=calls)
        if not isinstance(decision,dict) or not isinstance(decision.get('verdict'),dict):
            raise ValueError('MALFORMED_VERDICT')
        policy_issues=inspect_verdict(stage,decision['verdict'],query,calls,registry)
        decision=dict(decision)
        decision['issues']=sorted(set(decision.get('issues',[])) | set(policy_issues))
        decision['passed']=decision.get('passed') is True and not decision['issues']
        return decision
    def make_ground(attempt, instruction):
        ground=sampler.sample(record,registry,scenario,seed+attempt)
        return asdict(ground),{'model':'deterministic_sampler','attempt':attempt}
    def ground_from(raw):
        return GroundTruth(raw['scenario_type'],raw['candidate_tools'],raw['target_tools'],Metadata(**raw['metadata']))
    stage1=run_stage(1,make_ground,lambda raw:validators.validate_ground(ground_from(raw),registry),
                     lambda raw:verify_stage(1,ground_from(raw)),max_refinements)
    if not store(1,'ground_truth',stage1): return result
    ground=ground_from(stage1['output'])
    def make_query(attempt, instruction):
        raw,_,usage=query_generator.generate(client,ground,registry,refinement=instruction)
        return raw,usage
    def check_query(raw):
        check=validators.validate_query(raw,ground)
        if check.valid and query_key(raw['user_query']) in training_queries:
            return validators.feedback(2,[('user_query','copies_training_query')],('semantic_alignment','argument_grounding'))
        return check
    stage2=run_stage(2,make_query,check_query,
                     lambda raw:verify_stage(2,ground,query=raw['user_query']),max_refinements)
    if not store(2,'backward_query',stage2): return result
    query=stage2['output']['user_query']
    def make_calls(attempt, instruction):
        raw,_,usage=call_generator.generate(client,query,ground.candidate_tools,registry,ground.metadata.dataset_version,refinement=instruction)
        return raw,usage
    def check_calls(raw):
        check=validators.validate_calls(raw,ground,registry)
        critical={'format','x-ref','x-match','oneOf','anyOf','allOf'}
        unresolved={c['name'] for c in check.checks if c.get('status')=='not_verified'} & critical
        if unresolved:
            return validators.feedback(3,[issue('UNSUPPORTED_SCHEMA_CONSTRAINT',','.join(sorted(unresolved)),retryable=False)])
        return check
    stage3=run_stage(3,make_calls,check_calls,
                     lambda raw:verify_stage(3,ground,query=query,calls=raw['tool_calls']),max_refinements,redact_terms=ground.target_tools)
    if not store(3,'forward_calls',stage3): return result
    # Every preceding phase and every terminal deterministic/semantic decision must exist.
    if not all(result['stage_history'].get(k) and result['stage_history'][k][-1].get('semantic',{}).get('passed') is True for k in ('ground_truth','backward_query','forward_calls')):
        result['failure_reason']='incomplete_audit';return result
    result['status']='verified_candidate'
    return result

def generate(train_data, registry, client, limit, seed, with_report=False, verifier=None, max_refinements=3):
    if not isinstance(train_data, TrainOnlyDataset): raise TypeError('verified TrainOnlyDataset required')
    if not 1 <= limit <= 10: raise ValueError('limit must be 1 to 10')
    records=list(train_data.iter_records())
    random.Random(seed).shuffle(records)
    selected=[]
    eligible=0
    excluded=Counter()
    fallbacks=Counter()
    for record in records:
        try:
            sampler.sample(record,registry,'simple',seed)
        except ValueError as exc:
            excluded[str(exc)] += 1
            continue
        eligible += 1
        if len(selected) >= limit:
            continue
        scenario='simple' if len(selected)%2==0 else 'multiple'
        if scenario == 'multiple':
            try: sampler.sample(record,registry,scenario,seed+len(selected))
            except ValueError as exc:
                fallbacks[str(exc)] += 1
                scenario='simple'
        selected.append((record,scenario))
    train_queries={query_key(record['audio_text']) for record in records}
    metadata=None
    if verifier is not None:
        verifier_client=getattr(verifier,'client',None)
        if verifier_client is None or getattr(verifier_client,'model',None) == getattr(client,'model',None):
            raise ValueError('generator and verifier models must differ')
        client.keep_alive=0
        verifier_client.keep_alive=0
        client.prompt_byte_limit=max(1,client.options["num_ctx"]-client.options["num_predict"]-256)
        verifier_client.prompt_byte_limit=max(1,verifier_client.options["num_ctx"]-verifier_client.options["num_predict"]-256)
        generator_digest=client.model_digest()
        verifier_digest=verifier_client.model_digest()
        if not generator_digest or not verifier_digest or generator_digest==verifier_digest:
            raise ValueError('distinct local model digests required')
        client.unload()
        verifier_client.unload()
        metadata={'generator':client.model,'generator_digest':generator_digest,'generator_options':client.options,
                  'verifier':verifier_client.model,'verifier_digest':verifier_digest,'verifier_options':verifier_client.options,
                  'generator_keep_alive':client.keep_alive,'verifier_keep_alive':verifier_client.keep_alive,'verifier_think':getattr(verifier_client,'think',None)}
    results=[]
    for i,(record,scenario) in enumerate(selected):
        result=run_verified_attempt(record,registry,client,scenario,seed+i,train_queries,verifier,max_refinements,metadata) if verifier is not None else run_attempt(record,registry,client,scenario,seed+i,train_queries)
        results.append(result)
        if result.get('failure_reason')=='TransportError' or any(problem.get('error_code') in ('OUT_OF_MEMORY','TIMEOUT','TRANSPORT_FAILURE') for history in result.get('stage_history',{}).values() for step in history for problem in step.get('issues',[])):
            break

    report={'requested':limit,'eligible':eligible,'attempted':len(results),
            'shortfall':max(0,limit-len(results)), 'excluded_by_reason':dict(sorted(excluded.items())),
            'multiple_fallbacks_by_reason':dict(sorted(fallbacks.items()))}
    return (results,report) if with_report else results
