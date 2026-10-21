#!/usr/bin/env python3
"""Summarize accepted, complete interleaved runs without mixing screening data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument('folder')
ap.add_argument('--baseline', default='base')
ap.add_argument('--candidate', default='q6gpu')
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--early-stopped', action='store_true')
a = ap.parse_args()
folder = Path(a.folder)
events = [json.loads(line) for line in (folder/'runs.jsonl').read_text().splitlines()]
if a.early_stopped:
    assert (folder/'early-stop.json').exists() and events[-1]['kind'] == 'released'
else:
    assert any(row['kind'] == 'complete' for row in events), 'run did not complete'
complete_blocks = {(row['config'], row['round'], row['block_attempt']) for row in events if row['kind'] == 'block_complete'}
rows = [row for row in events if row['kind'] == 'request' and row['accepted'] and
        (row['config'], row['round'], row['block_attempt']) in complete_blocks]
keys = [(row['config'], row['round'], row['prompt']) for row in rows]
assert len(set(keys)) == len(keys), 'duplicate accepted request'
assert all(row['load_start'] < 2 and row['foreign_cores'] < 1 for row in rows)
names = [a.baseline, a.candidate]
prompts = sorted({row['prompt'] for row in rows})
by_key = {(row['config'], row['round'], row['prompt']): row for row in rows}
rounds = sorted({row['round'] for row in rows})
assert a.rounds >= 1 and len(rounds) == a.rounds and len(prompts) == 8
assert len(rows) == len(names)*len(rounds)*len(prompts)

def stats(values):
    return dict(n=len(values), median=st.median(values), minimum=min(values), maximum=max(values))

report = dict(categories={}, prompts={}, pairs=[], contamination_retries=sum(
    row['kind'] == 'request' and not row['accepted'] for row in events))
report['adoption_eligible_protocol'] = a.rounds >= 3 and not a.early_stopped
report['discarded_blocks'] = sum(row['kind'] == 'block_discarded' for row in events)
for category in ('code', 'en'):
    report['categories'][category] = {}
    for name in names:
        group = [r for r in rows if r['config'] == name and r['prompt'].startswith(category+'-')]
        assert len(group) == 4*a.rounds
        report['categories'][category][name] = dict(
            stream=stats([r['decode_tok_s'] for r in group]),
            server=stats([r['timings']['predicted_per_second'] for r in group]),
            tokens=sum(r['tokens'] for r in group),
            draft_generated=sum(r['timings']['draft_n'] for r in group),
            draft_accepted=sum(r['timings']['draft_n_accepted'] for r in group),
            foreign_cores=stats([r['foreign_cores'] for r in group]))
    for metric in ('stream', 'server'):
        old = report['categories'][category][a.baseline][metric]['median']
        new = report['categories'][category][a.candidate][metric]['median']
        report['categories'][category][metric+'_gain_pct'] = 100*(new/old-1)

for prompt in prompts:
    report['prompts'][prompt] = {}
    for name in names:
        group = [r for r in rows if r['config'] == name and r['prompt'] == prompt]
        report['prompts'][prompt][name] = dict(
            stream=stats([r['decode_tok_s'] for r in group]),
            server=stats([r['timings']['predicted_per_second'] for r in group]),
            hashes=dict(Counter(r['sha256'] for r in group)))
    for rnd in rounds:
        old, new = (by_key[(name, rnd, prompt)] for name in names)
        report['pairs'].append(dict(prompt=prompt, round=rnd,
            same_output=old['sha256'] == new['sha256'], tokens=[old['tokens'],new['tokens']],
            draft_generated=[old['timings']['draft_n'],new['timings']['draft_n']],
            draft_accepted=[old['timings']['draft_n_accepted'],new['timings']['draft_n_accepted']],
            stream_ratio=new['decode_tok_s']/old['decode_tok_s'],
            server_ratio=new['timings']['predicted_per_second']/old['timings']['predicted_per_second']))
report['same_output_pairs'] = sum(row['same_output'] for row in report['pairs'])
report['total_pairs'] = len(report['pairs'])
report['wall_seconds'] = events[-1]['epoch']-events[0]['epoch']
report['accepted_request_wall_seconds'] = sum(row['wall_s'] for row in rows)
report['paired_server_ratios'] = {
    category: stats([r['server_ratio'] for r in report['pairs'] if r['prompt'].startswith(category+'-')])
    for category in ('code','en')}
(folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('categories','same_output_pairs','total_pairs','wall_seconds','paired_server_ratios')}, indent=2))
