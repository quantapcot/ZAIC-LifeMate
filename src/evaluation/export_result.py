#!/usr/bin/env python3
"""Export completed persistent sessions without resetting/replaying them."""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'public_test'))
from zaic_tools import schema_validate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'submission.jsonl')
    args=parser.parse_args()
    items=[json.loads(line) for line in (ROOT/'public_test/public_test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    catalog=json.loads((ROOT/'public_test/public_test_tools_catalog.json').read_text(encoding='utf-8'))
    rows=[];missing=[];errors=[]
    for item in items:
        path=ROOT/'outputs'/'work_sessions'/(item['id']+'.pickle')
        if not path.exists():missing.append(item['id']);continue
        with path.open('rb') as stream:session=pickle.load(stream)
        calls=session.get_tool_calls()
        allowed={t for app in item['context']['installed_apps'].values() for t in app['tools']}|{'sys.ask_missing_info'}
        for index,result in enumerate(schema_validate(calls,catalog,allowed)):
            if not result['ok']:errors.append([item['id'],index,result])
        state_hash=session.get_state_hash()
        if not re.fullmatch('[0-9a-f]{64}',state_hash):errors.append([item['id'],'bad hash'])
        rows.append({'id':item['id'],'tool_calls':calls,'state_hash':state_hash})
    if missing or errors:
        raise SystemExit(json.dumps({'missing':missing,'errors':errors},ensure_ascii=False))
    assert len(rows)==len(items)==len({row['id'] for row in rows})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n' for row in rows),encoding='utf-8')
    reread=[json.loads(line) for line in args.output.read_text(encoding='utf-8').splitlines()]
    assert reread==rows
    report={'rows':len(rows),'unique_ids':len(set(row['id'] for row in rows)),
            'tool_calls':sum(len(row['tool_calls']) for row in rows),'schema_errors':len(errors),
            'empty_tool_sequences':sum(not row['tool_calls'] for row in rows),'utf8_jsonl_round_trip':True,
            'state_hash_source':'get_state_hash() from the original persistent ToolEnv session',
            'accuracy_measured':False}
    args.output.with_suffix('.validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
