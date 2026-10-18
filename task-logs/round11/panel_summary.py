#!/usr/bin/env python3
"""Archive only this round's panel rows and retain context/quiet failures."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11/panels'
HISTORY = Path('/opt/llm-gateway/data/bench-home-keep75ud/data/bench/results.jsonl')


def main():
    rows = [json.loads(line) for line in HISTORY.read_text().splitlines()]
    rows = [r for r in rows if r.get('label') == 'card-round11-2026-10-18' and r.get('model') == 'ling-round11']
    (OUT / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    groups = collections.defaultdict(list)
    for row in rows:
        if row['kind'] in ['hard', 'agent']:
            groups[row['kind']].append(row)
    result = {}
    for kind, group in groups.items():
        counts = collections.Counter((r['run'], r['id']) for r in group)
        assert max(counts.values()) == 1, 'duplicate panel observations'
        categories = collections.defaultdict(list)
        for row in group:
            categories[row['category']].append(row)
        result[kind] = dict(right=sum(r['right'] for r in group), total=len(group),
            by_category={c: dict(right=sum(r['right'] for r in g), total=len(g)) for c, g in categories.items()},
            failures=[r for r in group if not r['right']])
    events = [json.loads(line) for line in (OUT / 'runs.jsonl').read_text().splitlines()]
    chats = [r for r in events if r['kind'] == 'panel_request']
    responses = [json.loads(line) for line in (OUT / 'responses.jsonl').read_text().splitlines()]
    assert len(chats) == len(responses), 'request archive is incomplete'
    foreign_failures = []
    for event, response in zip(chats, responses):
        if not event['quiet']:
            request = response['request']
            foreign_failures.append(dict(event=event, seed=request.get('seed'),
                first_message=str(request.get('messages', [{}])[0].get('content', ''))[:240],
                scope='Correctness outcome retained; this request is not a clean timing measurement.'))
    result.update(complete=len(groups['hard']) == 42 and len(groups['agent']) == 36,
        chat_requests=len(chats), quiet_requests=sum(r['quiet'] for r in chats),
        max_foreign_cores=max((r['foreign_cores'] for r in chats), default=None),
        foreign_load_failures=foreign_failures,
        historical_controls=json.loads((ROOT / 'task-logs/round10/panel-controls.json').read_text()),
        scope='Three seeds, model-card sampling and thinking on. Historical controls are not paired runtime measurements. Context is 32768; long-combine remains in the denominator.')
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'historical_controls'}, indent=2))


if __name__ == '__main__':
    main()
