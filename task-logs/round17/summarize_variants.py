#!/usr/bin/env python3
"""Summarize only completed, uncontaminated blocks and compare paired outputs."""
import collections
import json
from pathlib import Path
import statistics
import sys

p = Path(sys.argv[1])
if (p / 'INVALID.json').exists():
    raise SystemExit('Excluded run: ' + (p / 'INVALID.json').read_text())
events = [json.loads(line) for line in (p / 'runs.jsonl').read_text().splitlines()]
complete = {(e['block'], e['attempt']) for e in events if e['kind'] == 'block_complete'}
rows = [e for e in events if e['kind'] == 'request' and (e['block'], e['attempt']) in complete]
assert all(e['accepted'] for e in rows)
groups = collections.defaultdict(list)
for e in rows:
    groups[(e['config'], e['prompt'], e['phase'])].append(e)
report = {'complete_blocks': sorted(complete), 'comparisons': []}
for prompt, phase in sorted({(e['prompt'], e['phase']) for e in rows}):
    a, b = groups['A', prompt, phase], groups['B', prompt, phase]
    if not a or not b:
        continue
    def signature(e):
        t = e['timings']
        return (e['sha256'], e['tokens'], t['draft_n'], t['draft_n_accepted'], t['cache_n'], t['prompt_n'])
    sa, sb = {signature(e) for e in a}, {signature(e) for e in b}
    ma, mb = [statistics.mean(e['decode_tok_s'] for e in group) for group in (a, b)]
    report['comparisons'].append(dict(prompt=prompt, phase=phase, n_a=len(a), n_b=len(b),
        a_tok_s=ma, b_tok_s=mb, gain_pct=(mb / ma - 1) * 100,
        outputs_and_drafts_identical=sa == sb and len(sa) == 1,
        a_signatures=sorted(sa), b_signatures=sorted(sb),
        a_tok_s_values=[e['decode_tok_s'] for e in a], b_tok_s_values=[e['decode_tok_s'] for e in b]))
report['gpu_memory_mib'] = {name: sorted({mib for e in rows if e['config'] == name for mib in e['gpu'].values()}) for name in ('A', 'B')}
report['foreign_cores_max'] = max((e['foreign_cores'] for e in rows), default=None)
report['note'] = 'Client decode=(completion_tokens-1)/(last-first content chunk). No instrumented requests. Only completed blocks included.'
(p / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
