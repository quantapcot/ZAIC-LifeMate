"""Three bounded stages; validation is preliminary and has no side effects."""
import random
import time
from dataclasses import asdict
from . import sampler, query_generator, call_generator, validators
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
        result['failure_stage']=1;result['failure_reason']=str(exc);return result
    try:
        raw,query,usage=query_generator.generate(client,ground,registry)
        check=validators.validate_query(raw,ground)
        if check.valid and query_key(query.user_query) in training_queries:
            check=validators.feedback(2, [('user_query','copies_training_query')], ('semantic_alignment', 'argument_grounding'))
        result['stages']['backward_query']={'output':asdict(query),'model_output':raw,'validation':asdict(check),'usage':usage,'latency_seconds':usage['latency_seconds']}
        if not check.valid: raise ValueError('invalid_query')
    except (TransportError,InvalidModelOutput,ValueError) as exc:
        result['failure_stage']=2;result['failure_reason']=type(exc).__name__ if isinstance(exc,(TransportError,InvalidModelOutput)) else 'invalid_query';return result
    try:
        raw,calls,usage=call_generator.generate(client,query.user_query,ground.candidate_tools,registry,ground.metadata.dataset_version)
        check=validators.validate_calls(raw,ground,registry)
        result['stages']['forward_calls']={'output':asdict(calls) if calls else None,'model_output':raw,'validation':asdict(check),'usage':usage,'latency_seconds':usage['latency_seconds']}
        if not check.valid: raise ValueError('invalid_calls')
    except (TransportError,InvalidModelOutput,ValueError) as exc:
        result['failure_stage']=3;result['failure_reason']=type(exc).__name__ if isinstance(exc,(TransportError,InvalidModelOutput)) else 'invalid_calls';return result
    result['status']='provisional'
    return result


def generate(train_data, registry, client, limit, seed):
    if not isinstance(train_data, TrainOnlyDataset): raise TypeError('verified TrainOnlyDataset required')
    if not 1 <= limit <= 10: raise ValueError('limit must be 1 to 10')
    records=list(train_data.iter_records())
    random.Random(seed).shuffle(records)
    # Each selected record gets one attempt, no semantic or technical retries.
    selected=[]
    for record in records:
        scenario='simple' if len(selected)%2==0 else 'multiple'
        try: sampler.sample(record,registry,scenario,seed+len(selected))
        except ValueError: continue
        selected.append((record,scenario))
        if len(selected)==limit: break
    train_queries={query_key(record['audio_text']) for record in records}
    return [run_attempt(record,registry,client,scenario,seed+i,train_queries) for i,(record,scenario) in enumerate(selected)]
