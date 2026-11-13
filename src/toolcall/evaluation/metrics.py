"""Structural diagnostics; not an implementation of the official score."""
from collections import Counter
import json
import math
import statistics


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
    return {'sample_count':n,'tool_name_sequence_accuracy':div(names,n),'call_sequence_exact_match':div(exact,n),'aligned_argument_exact_match':div(args_ok,args_n),'aligned_argument_count':args_n,'call_precision':precision,'call_recall':recall,'call_f1':f1,'missing_actions_per_gold_action':div(missing,gold_n),'extra_actions_per_gold_action':div(extra,gold_n),'missing_info_binary_accuracy':div(missing_correct,n),'empty_gold_empty_prediction':div(empty_correct,empty_n),'empty_gold_count':empty_n,'invalid_proposals':sum(r['invalid_proposals'] for r in predictions),'sdk_validation_success_rate':div(sum(r['sdk_valid'] for r in predictions),sum(r['sdk_checks'] for r in predictions)),'execution_reported_success_rate':div(sum(r['execution_success'] for r in predictions),sum(r['executions'] for r in predictions)),'tool_calls_per_sample':div(pred_n,n),'final_state_hash_match':None,'official_score':None,'task_completion':None,'scenario_exact_match':{k:{'count':len(v),'rate':div(sum(v),len(v))} for k,v in groups.items()},'unknown_scenarios':['dependent','missing_information_cause','out_of_scope','context_dependent','history_dependent'],'latency_mean':statistics.mean(times) if times else None,'latency_p50':statistics.median(times) if times else None,'latency_p95':times[max(0,math.ceil(.95*len(times))-1)] if times else None,'model_seconds_total':sum(r['model_seconds'] for r in predictions),'tool_seconds_total':sum(r['tool_seconds'] for r in predictions),'output_tokens_total':sum(r['output_tokens'] for r in predictions),'timeout_rate':div(sum(r['error']=='timeout' for r in predictions),n),'model_failure_rate':div(sum(r['error']=='model_failure' for r in predictions),n),'error_analysis':{'wrong_tool_sequence':wrong_tool,'aligned_wrong_argument':wrong_argument,'missing_actions':missing,'extra_actions':extra,'other_categories':'unknown'}}
