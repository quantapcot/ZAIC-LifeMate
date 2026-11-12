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


def generate(train_data, registry, client, limit, seed, with_report=False):
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
    results=[run_attempt(record,registry,client,scenario,seed+i,train_queries) for i,(record,scenario) in enumerate(selected)]
    report={'requested':limit,'eligible':eligible,'attempted':len(results),
            'shortfall':max(0,limit-len(results)), 'excluded_by_reason':dict(sorted(excluded.items())),
            'multiple_fallbacks_by_reason':dict(sorted(fallbacks.items()))}
    return (results,report) if with_report else results
