#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def batch(pos, ids, prompt):
    return dict(version=2, context=1, ubatch=pos, n_groups=2, n_tokens=len(ids), prefill=prompt,
                prefill_end=prompt, positions=list(range(pos, pos + len(ids))), seq_ids=[[0] for _ in ids],
                layers=[dict(layer=0, group=0, slots=1, n_expert=4, expert_bytes=100, cache=not prompt,
                             rows=list(range(len(ids))), ids=[[i] for i in ids]),
                        dict(layer=1, group=-1, slots=0, n_expert=4, expert_bytes=200, cache=False,
                             rows=list(range(len(ids))), ids=[[0] for _ in ids])])


def main():
    path = OUT / 'metric-fixture.jsonl'
    path.write_text('\n'.join(json.dumps(b) for b in [batch(0, [0], True), batch(1, [0, 1, 1, 2], False)]) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_MOE_')}
    env['GGML_MOE_REPLAY_MISS_HISTOGRAM'] = '1'
    binary = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'build-cpu/bin/test-moe-cache')
    result = subprocess.check_output([binary, '--replay-log', str(path), '0', '1', '1', '0', '128'], env=env, text=True)
    (OUT / 'metric-check.log').write_text(result)
    metrics = dict(re.findall(r'([\w/]+)=([^\s]+)', next(l for l in result.splitlines() if l.startswith('context='))))
    assert metrics['all_byte_hit_rate'] == '6.6667%', metrics
    assert metrics['decode_byte_hit_rate'] == '8.3333%', metrics
    expected = {(0, 1): 1, (0, 2): 1, (1, 4): 1}
    actual = {}
    for line in result.splitlines():
        if line.startswith('miss_histogram'):
            row = dict(re.findall(r'(\w+)=(\d+)', line))
            actual[int(row['layer']), int(row['multiplicity'])] = int(row['experts'])
    assert actual == expected, actual
    print('Weighted prefill/decode metrics and missed-expert token histogram passed.')


if __name__ == '__main__':
    main()
