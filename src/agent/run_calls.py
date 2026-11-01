#!/usr/bin/env python3
"""Execute reviewed calls through one persistent ToolEnv per public sample.

Sessions are pickled after every call, so a follow-up batch resumes the same
session instead of resetting the simulator. Only tool return values are exposed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    session_dir = ROOT / 'outputs' / 'work_sessions'
    session_dir.mkdir(exist_ok=True)
    results = []
    data = json.loads(args.batch.read_text(encoding='utf-8'))
    for row in data:
        sample_id = row['id']
        path = session_dir / f'{sample_id}.pickle'
        if path.exists():
            with path.open('rb') as stream:
                session = pickle.load(stream)
        else:
            session = ToolEnv('public_test', sample_id, root=ROOT)
        outputs = []
        for call in row.get('tool_calls', []):
            name, arguments = call['name'], call.get('arguments') or {}
            check = session.validate(name, arguments)
            if not check['ok']:
                raise ValueError(f'{sample_id}: {name}: {check}')
            result = session.call(name, arguments)
            outputs.append({'call': call, 'result': result})
            with path.open('wb') as stream:
                pickle.dump(session, stream)
        if not path.exists():
            with path.open('wb') as stream:
                pickle.dump(session, stream)
        results.append({'id': sample_id, 'outputs': outputs,
                        'tool_calls': session.get_tool_calls(),
                        'state_hash': session.get_state_hash()})
    text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
