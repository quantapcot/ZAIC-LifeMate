"""Structural diagnostics; not an implementation of the official score."""
from collections import Counter
import json
import math
import statistics
from src.toolcall.evaluation.routing_diagnostics import aligned_recall


def metrics(predictions, labels):
    n = len(predictions); exact = names = args_ok = args_n = tp = pred_n = gold_n = 0
    missing = extra = 0; groups = {'zero': [], 'single': [], 'multi': []}; wrong_tool = wrong_argument = 0
    missing_correct = empty_correct = empty_n = 0
    for row in predictions:
        p = row['tool_calls']; g = labels[row['id']]['tool_calls']
        pn = [c['name'] for c in p]; gn = [c['name'] for c in g]
        name_match = pn == gn; names += name_match; exact += p == g
        wrong_tool += not name_match
        for pc,gc in zip(p,g):
            if pc['name'] == gc['name']:
                args_n += 1; args_ok += pc['arguments'] == gc['arguments']; wrong_argument += pc['arguments'] != gc['arguments']
        encode = lambda calls: Counter(json.dumps(c,sort_keys=True,ensure_ascii=False) for c in calls)
        a,b = encode(p),encode(g); tp += sum((a & b).values()); pred_n += len(p); gold_n += len(g)
        ca,cb = Counter(pn),Counter(gn); missing += sum((cb-ca).values()); extra += sum((ca-cb).values())
        missing_correct += (('sys.ask_missing_info' in pn) == ('sys.ask_missing_info' in gn))
        if not g: empty_n += 1; empty_correct += not p
        groups['zero' if not g else 'single' if len(g)==1 else 'multi'].append(p == g)
    div = lambda a,b: a/b if b else None
    precision,recall=div(tp,pred_n),div(tp,gold_n)
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0.0 if precision is not None and recall is not None else None)
    times=sorted(r['latency_seconds'] for r in predictions)
    report = {'sample_count':n,'tool_name_sequence_accuracy':div(names,n),'call_sequence_exact_match':div(exact,n),'aligned_argument_exact_match':div(args_ok,args_n),'aligned_argument_count':args_n,'call_precision':precision,'call_recall':recall,'call_f1':f1,'missing_actions_per_gold_action':div(missing,gold_n),'extra_actions_per_gold_action':div(extra,gold_n),'missing_info_binary_accuracy':div(missing_correct,n),'empty_gold_empty_prediction':div(empty_correct,empty_n),'empty_gold_count':empty_n,'invalid_proposals':sum(r['invalid_proposals'] for r in predictions),'sdk_validation_success_rate':div(sum(r['sdk_valid'] for r in predictions),sum(r['sdk_checks'] for r in predictions)),'execution_reported_success_rate':div(sum(r['execution_success'] for r in predictions),sum(r['executions'] for r in predictions)),'tool_calls_per_sample':div(pred_n,n),'final_state_hash_match':None,'official_score':None,'task_completion':None,'scenario_exact_match':{k:{'count':len(v),'rate':div(sum(v),len(v))} for k,v in groups.items()},'unknown_scenarios':['dependent','missing_information_cause','out_of_scope','context_dependent','history_dependent'],'latency_mean':statistics.mean(times) if times else None,'latency_p50':statistics.median(times) if times else None,'latency_p95':times[max(0,math.ceil(.95*len(times))-1)] if times else None,'model_seconds_total':sum(r['model_seconds'] for r in predictions),'tool_seconds_total':sum(r['tool_seconds'] for r in predictions),'output_tokens_total':sum(r['output_tokens'] for r in predictions),'timeout_rate':div(sum(r['error']=='timeout' for r in predictions),n),'model_failure_rate':div(sum(r['error']=='model_failure' for r in predictions),n),'error_analysis':{'wrong_tool_sequence':wrong_tool,'aligned_wrong_argument':wrong_argument,'missing_actions':missing,'extra_actions':extra,'other_categories':'unknown'}}

    # Gold is consulted only here, after the inference stage. Candidate recall is
    # per gold call, against the union of candidate sets seen during that sample.
    hits = targets = first_hits = first_targets = premature = first_correct = wrong_selected = 0
    aligned_hits=aligned_total=0
    stops = Counter(); taxonomy = Counter(); internal = sdk_rejects = proposed = candidate_counts = steps = 0
    for row in predictions:
        gold=labels[row['id']]['tool_calls']; trace=row.get('trace',[])
        ah,at=aligned_recall(trace,gold);aligned_hits+=ah;aligned_total+=at
        seen=set(name for step in trace for name in step.get('candidate_tools',[]))
        first=set(trace[0].get('candidate_tools',[])) if trace else set()
        targets += len(gold); hits += sum(c['name'] in seen for c in gold)
        taxonomy['TOOL_NOT_RETRIEVED'] += sum(c['name'] not in seen for c in gold)
        if gold:
            first_targets+=1; first_hits+=gold[0]['name'] in first
            calls=[x for x in trace if x.get('model_action')=='call']
            if calls:
                first_correct+=calls[0].get('proposed_tool')==gold[0]['name']
                wrong_selected+=(gold[0]['name'] in first and calls[0].get('proposed_tool')!=gold[0]['name'])
        for step in trace:
            steps+=1; candidate_counts+=step.get('candidate_count',0)
            internal+=step.get('internal_validation')=='rejected'; sdk_rejects+=step.get('sdk_validation')=='rejected'
            for issue in step.get('validation_errors',[]): taxonomy[issue.get('code','unknown')]+=1
            taxonomy['SDK_VALIDATION_FAILED'] += step.get('sdk_validation')=='rejected'
            status=step.get('sdk_result_status') or {}
            taxonomy['TOOL_EXECUTION_FAILED'] += step.get('executed',False) and status.get('ok') is False
            proposed+=step.get('model_action')=='call'
        stop=trace[-1].get('stop_reason') if trace else 'unknown'; stops[stop or 'unknown']+=1
        if stop=='FINISH' and len(row['tool_calls'])<len(gold): premature+=1
        if stop in ('REPEATED_INEFFECTIVE_CALL','PROMPT_BUDGET_EXCEEDED','MODEL_OUTPUT_FAILURE','STEP_LIMIT','TIMEOUT'): taxonomy[stop]+=1
    taxonomy['WRONG_TOOL_SELECTION_FIRST_WHEN_RETRIEVED']=wrong_selected
    taxonomy['PREMATURE_FINISH_COUNT_HEURISTIC']=premature
    report.update(step_aligned_candidate_recall=div(aligned_hits,aligned_total),step_aligned_observations=aligned_total,failure_taxonomy=dict(taxonomy),candidate_recall_union=div(hits,targets),candidate_miss_rate=div(targets-hits,targets),first_gold_tool_recall_initial=div(first_hits,first_targets),candidate_targets=targets,first_action_tool_accuracy=div(first_correct,first_targets),first_target_retrieved_but_other_tool_proposed=wrong_selected,average_candidates_per_step=div(candidate_counts,steps),internal_rejections=internal,sdk_rejections=sdk_rejects,model_proposals=proposed,invalid_proposal_rate=div(sum(r['invalid_proposals'] for r in predictions),proposed),stop_reasons=dict(stops),premature_finish_by_count=div(premature,n),model_calls_per_sample=div(sum(r.get('model_calls',0) for r in predictions),n),router_seconds_total=sum(r.get('router_seconds',0) for r in predictions),prompt_budget_failures=sum(r.get('error')=='prompt_budget' for r in predictions))
    return report
