#!/usr/bin/env python3
"""Keep numerical checks and quiet-machine acceptance distinct."""
import filecmp
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'


def summarize():
    historical = json.loads((ROOT / 'task-logs/round9/quality-summary.json').read_text())
    reference = next(r for r in historical['perplexity'] if r['name'] == 'ling-off-compaction0-ppl')
    rows = []
    for name in ['ling-off-ppl', 'ling-r7-4608-ppl']:
        folder = OUT / name
        events = [json.loads(line) for line in (folder / 'runs.jsonl').read_text().splitlines()]
        gate = next(r for r in reversed(events) if r['kind'] == 'quality')
        log = folder / f"attempt-{gate['attempt']}.log"
        values = re.findall(r'Final estimate: PPL = ([\d.]+)', log.read_text())
        assert values, 'missing perplexity result'
        rows.append(dict(name=name, ppl=float(values[-1]), quiet_gate=gate['accepted'],
            foreign_cores=gate['foreign_cores'], log=str(log)))
    baseline = float(rows[0]['ppl'])
    for row in rows:
        value = float(row['ppl'])
        row.update(relative_percent=100*(value/baseline-1), numerical_gate=value <= baseline*1.005)
    control = next((ROOT / 'task-logs/round9/ling-off-probe').glob('*.logits.bin'))
    candidate = next((OUT / 'ling-off-knobs-probe').glob('*.logits.bin'))
    probe_events = [json.loads(line) for line in (candidate.parent / 'runs.jsonl').read_text().splitlines()]
    probe_gate = next(r for r in reversed(probe_events) if r['kind'] == 'quality')
    probe = dict(control=str(control), candidate=str(candidate), bytes=candidate.stat().st_size,
        bit_exact=filecmp.cmp(control, candidate, shallow=False),
        sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), quiet_gate=probe_gate['accepted'],
        foreign_cores=probe_gate['foreign_cores'])
    return dict(perplexity=rows, probe=probe, historical_baseline=reference,
        historical_cache=next(r for r in historical['perplexity'] if r['name'] == 'ling-slots-4608-dspark-ppl'),
        fresh_baseline_matches_historical=baseline == reference['ppl'],
        numerical_gate=probe['bit_exact'] and all(r['numerical_gate'] for r in rows),
        quiet_gate=probe['quiet_gate'] and all(r['quiet_gate'] for r in rows),
        scope='Perplexity and logits probe exercise the target/cache kernels. They do not run the speculative drafter; actual speculative behavior is checked by response hashes and gateway panels. A failed foreign-load gate is not silently relabeled as a clean measurement.')


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2))
