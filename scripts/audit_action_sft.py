"""Aggregate TRAIN-only data and official first-call SDK validation audit."""
import argparse
import collections
import json
import time
from scripts.evaluate_action_agent import ROOT,WORKTREE
from src.toolcall.agent.action_agent import available_tools,basic_issues,delegated_constraints
from src.toolcall.agent.tool_router import tokens
from src.toolcall.agent.toolenv_adapter import open_train_env
from src.toolcall.data.schema_registry import ToolRegistry
from src.toolcall.evaluation.heldout import Heldout


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);a=parser.parse_args()
    out=(WORKTREE/a.output_dir).resolve();base=(WORKTREE/'outputs/action_sft').resolve()
    if base not in out.parents:raise ValueError('output must be under outputs/action_sft')
    d=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train');reg=ToolRegistry.from_catalog(ROOT/'train/train_tools_catalog.json');d.finish_predictions();labels=d.labels(d.ids)
    out.mkdir(parents=True,exist_ok=False);c=collections.Counter();flags=collections.Counter();t=time.perf_counter()
    for key in sorted(d.ids):
        row=d.records[key];label=labels[key];calls=label.get('tool_calls');results=label.get('tool_results');c['records']+=1
        if not isinstance(calls,list) or not isinstance(results,list):c['malformed_label']+=1;continue
        c['zero' if not calls else 'single' if len(calls)==1 else 'multi']+=1
        c['result_count_aligned' if len(calls)==len(results) else 'result_count_misaligned']+=1
        c['result_dicts']+=sum(isinstance(x,dict) for x in results)
        c['missing_info_records']+=any(isinstance(x,dict) and x.get('name')=='sys.ask_missing_info' for x in calls)
        if not calls:
            c['zero_with_history']+=bool(row.get('history'));c['zero_empty_transcript']+=not bool(row.get('audio_text','').strip())
            c['zero_action_verb_heuristic']+=bool(set(tokens(row.get('audio_text',''))) & {'bat','tat','gui','goi','dat','tao','xoa','hen','nhac','them','tim','mo','luu'})
        names=available_tools(row.get('context',{}),reg)
        for call in calls:
            if not isinstance(call,dict) or not isinstance(call.get('name'),str) or not isinstance(call.get('arguments'),dict):c['malformed_call']+=1;continue
            if call['name'] not in reg.list_tools():c['unknown_tool_call']+=1;continue
            if call['name'] not in names:c['unavailable_tool_call']+=1
            if basic_issues(call['arguments'],reg.get_schema(call['name'])):c['basic_invalid_call']+=1
        if calls and isinstance(calls[0],dict) and calls[0].get('name') in reg.list_tools():
            flags.update(delegated_constraints(reg.get_schema(calls[0]['name'])))
            try:
                env=open_train_env(ROOT,key);verdict=env.validate(calls[0]['name'],calls[0]['arguments']);c['sdk_first_accepted' if verdict.get('ok') is True else 'sdk_first_rejected']+=1
            except Exception:c['sdk_first_error']+=1
    report={'source_partition':'train','dataset_sha256':d.manifest['dataset_sha256'],'catalog_sha256':d.manifest['catalog_sha256'],'counts':dict(c),'first_schema_delegated_constraints':dict(flags),'result_alignment_note':'equal lengths; results do not identify tool names, so semantic/order alignment is not independently established','zero_semantics_note':'heuristic verb overlap is not a scope classification','seconds':time.perf_counter()-t}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
