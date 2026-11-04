#!/usr/bin/env python3
"""Print compact input packets, without opening environment state files."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--ids', nargs='*')
    p.add_argument('--full-history', action='store_true')
    p.add_argument('--brief',action='store_true')
    a = p.parse_args()
    items = [
        json.loads(line)
        for line in (ROOT / 'public_test/public_test.jsonl').read_text().splitlines()
    ]
    transcripts = {}
    for path in [
        ROOT / 'outputs/transcripts/public.jsonl',
        ROOT / 'outputs/transcripts/public_corrected.jsonl',
    ]:
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                transcripts[Path(row['path']).stem] = row['text']
    selected = [x for x in items if x['id'] in a.ids] if a.ids else items[a.start:a.start+a.count]
    for x in selected:
        history = []
        for turn in x['history']:
            if turn['role'] == 'assistant':
                if a.brief:
                    history.append(turn.get('text',''))
                elif a.full_history:
                    history.append(turn)
                else:
                    history.append({k: turn[k] for k in ('text', 'tool_calls', 'tool_results') if k in turn})
            elif a.full_history:
                history.append({'role':'user', 'text':transcripts.get(Path(turn['audio']).stem, '?')})
        if a.brief:
            print(json.dumps({'id':x['id'],'now':x['context']['current_time'],'apps':list(x['context']['installed_apps'])[:5],'request':transcripts.get(x['id'],'PENDING'),'history':history},ensure_ascii=False))
            continue
        print(json.dumps({'id':x['id'],'now':x['context']['current_time'],
                          'profile':x['context']['user_profile'],
                          'apps':list(x['context']['installed_apps']),
                          'request':transcripts.get(x['id'],'PENDING'),
                          'history':history},ensure_ascii=False))


if __name__ == '__main__':
    main()
