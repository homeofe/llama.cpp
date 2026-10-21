#!/usr/bin/env python3
"""Summarize completed clean native-reference blocks; preserve novel/repeat separately."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+', type=Path)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
rows = []
sources = []
for directory in a.runs:
    assert not (directory/'DISCARDED.json').exists(), f'discarded measurements in {directory}'
    events = [json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
    complete = {(e['config'],e['block_attempt']) for e in events if e['kind']=='block_complete'}
    assert complete, f'no complete block in {directory}'
    for config,attempt in sorted(complete):
        block = [e for e in events if e['kind']=='request' and (e['config'],e['block_attempt'])==(config,attempt)]
        assert len(block)==16 and len({(e['prompt'],e['phase']) for e in block})==16
        assert all(e['accepted'] and e['load_start']<2 and e['foreign_cores']<1 for e in block)
        for e in block:
            answer = directory/f"{config}-b{attempt}-{e['prompt']}-{e['phase']}.txt"
            assert hashlib.sha256(answer.read_bytes()).hexdigest()==e['sha256']
        rows.extend(block)
        sources.append(dict(directory=str(directory),config=config,attempt=attempt))
summary = {}
for config in sorted({e['config'] for e in rows}):
    summary[config] = {}
    for category in ['code','en']:
        for phase in ['novel','repeat']:
            group = [e for e in rows if e['config']==config and e['prompt'].startswith(category) and e['phase']==phase]
            value = dict(n=len(group),tokens=[e['tokens'] for e in group])
            for metric in ['server','stream']:
                numbers = [e['timings']['predicted_per_second'] if metric=='server' else e['decode_tok_s'] for e in group]
                value[metric] = dict(median=statistics.median(numbers),min=min(numbers),max=max(numbers))
            value['draft_accepted'] = sum(e['timings']['draft_n_accepted'] for e in group)
            value['draft_offered'] = sum(e['timings']['draft_n'] for e in group)
            summary[config][category+'-'+phase] = value
result = dict(protocol='sequential pilot, not interleaved adoption proof',sources=sources,summary=summary)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
