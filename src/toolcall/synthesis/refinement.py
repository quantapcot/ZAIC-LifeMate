"""Stage-specific generation, validation and bounded feedback, with local audit history."""
import json
import time
import re
from dataclasses import asdict
from src.toolcall.llm import TransportError, InvalidModelOutput

MAX_REFINEMENTS=3


def run_stage(stage, producer, deterministic, semantic, max_refinements=3, redact_terms=(), can_refine=None):
    if type(max_refinements) is not int or not 0 <= max_refinements <= MAX_REFINEMENTS:
        raise ValueError('max_refinements must be between 0 and 3')
    if semantic is None: raise ValueError('semantic verifier required')
    history=[]
    previous=None
    for attempt in range(max_refinements+1):
        entry={'stage':stage,'attempt':attempt,'refinement_instruction':None,'output':None,
               'deterministic':None,'semantic':None,'generation_usage':None}
        if attempt:
            issues=[dict(problem) for prior in history for problem in prior.get('issues',[])]
            for problem in issues:
                for key,value in problem.items():
                    if isinstance(value,str):
                        for term in redact_terms:
                            if term: value=re.sub(re.escape(term),'[tool]',value,flags=re.IGNORECASE)
                        problem[key]=value
            instruction={'objective':'Correct the last output using the official definitions and all reported issues; preserve accepted upstream inputs.',
                         'previous_output':previous,'issues':issues,'required_output':'original structured output schema'}
            # Refuse oversize feedback instead of silently dropping evidence.
            if len(json.dumps(instruction,ensure_ascii=False))>12000:
                entry['issues']=[{'error_code':'FEEDBACK_TOO_LARGE','retryable':False}];history.append(entry);break
            entry['refinement_instruction']=instruction
        else: instruction=None
        start=time.perf_counter()
        try:
            output,usage=producer(attempt,instruction)
            entry['output']=output;entry['generation_usage']=usage
            previous=output
        except (TransportError,InvalidModelOutput) as exc:
            entry['issues']=[{'error_code':exc.code,'field':exc.field,'source':'transport' if isinstance(exc,TransportError) else 'model_output',
                              'retryable':isinstance(exc,InvalidModelOutput) and exc.code!='RESPONSE_TOO_LARGE'}]
            if exc.raw_output is not None: entry['output']=exc.raw_output; previous=exc.raw_output
            entry['latency_seconds']=time.perf_counter()-start;history.append(entry)
            if not entry['issues'][0]['retryable']:break
            continue
        except ValueError as exc:
            entry['issues']=[{'error_code':str(exc),'source':'sampler' if stage==1 else 'generation_input','retryable':False}]
            if hasattr(exc,'budget'): entry['input_budget']=exc.budget
            entry['latency_seconds']=time.perf_counter()-start;history.append(entry);break
        check=deterministic(output)
        entry['deterministic']=asdict(check)
        if not check.valid:
            entry['issues']=check.issues
            entry['latency_seconds']=time.perf_counter()-start;history.append(entry)
            if not check.issues or not all(x.get('retryable',False) for x in check.issues): break
            if can_refine is not None:
                allowed,reason=can_refine(history)
                if not allowed: history[-1]['stop_reason']=reason;break
            continue
        try:
            decision=semantic(output)
            entry['semantic']=decision
        except (TransportError,InvalidModelOutput,ValueError) as exc:
            entry['issues']=[{'error_code':getattr(exc,'code','INVALID_VERDICT'),'source':'verifier_runtime','retryable':False}]
            if isinstance(exc,InvalidModelOutput) and exc.raw_output is not None: entry['verifier_invalid_output_excerpt']=exc.raw_output
            entry['latency_seconds']=time.perf_counter()-start;history.append(entry);break
        entry['latency_seconds']=time.perf_counter()-start
        if decision.get('passed') is True and not decision.get('issues') and decision.get('verdict',{}).get('verdict') == 'pass':
            entry['issues']=[];history.append(entry)
            return {'status':'passed','output':output,'history':history,'refinements':attempt}
        entry['issues']=[{'error_code':code,'source':'semantic_verifier','retryable':True} for code in decision.get('issues',[])]
        for check in decision.get('verdict',{}).get('checks',[]):
            if check.get('status')!='pass':
                entry['issues'].append({'error_code':check.get('error_code') or 'SEMANTIC_FAILURE','field':check.get('field',''),
                                        'message':check.get('message','')[:300], 'source':'semantic_verifier','retryable':True})
        history.append(entry)
        if decision.get('verdict',{}).get('verdict')=='uncertain':
            return {'status':'manual_review_required','output':None,'history':history,'refinements':attempt}
        if not entry['issues']: break
        if can_refine is not None:
            allowed,reason=can_refine(history)
            if not allowed: history[-1]['stop_reason']=reason;break
    return {'status':'rejected','output':None,'history':history,'refinements':max(0,len(history)-1)}
