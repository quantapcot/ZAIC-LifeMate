"""Offline retrieval diagnostics, never used by agent inference."""

def recall_at_k(ranked, gold, k):
    return sum(name in ranked[:k] for name in gold),len(gold)


def aligned_recall(trace, gold):
    """First candidate set at each exactly executed gold prefix; stop on divergence."""
    i=0;seen=set();hit=total=0
    for step in trace:
        if i>=len(gold): break
        if i not in seen:
            seen.add(i);total+=1;hit+=gold[i]['name'] in step.get('candidate_tools',[])
        if step.get('executed'):
            if {'name':step.get('proposed_tool'),'arguments':step.get('proposed_arguments')} != gold[i]: break
            i+=1
    return hit,total
