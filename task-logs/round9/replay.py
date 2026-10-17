#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'
TRACES = Path('/opt/llm-gateway/data/moe-quant/traces')
PROFILE = '/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1-2048mib.profile'
POLICIES = [
    ('r7', 128, 32, 1024, 0, 0, 0),
    ('decay64', 64, 32, 1024, 0, 0, 0),
    ('decay256', 256, 32, 1024, 0, 0, 0),
    ('prefill512', 128, 32, 512, 0, 0, 0),
    ('admit16', 128, 16, 1024, 0, 0, 0),
    ('admit64', 128, 64, 1024, 0, 0, 0),
    ('byte', 128, 32, 1024, 1, 0, 0),
    ('byte-decay64', 64, 32, 1024, 1, 0, 0),
    ('byte-soft100', 128, 32, 1024, 1, 100, 0),
    ('byte-soft150', 128, 32, 1024, 1, 150, 0),
    ('byte-prefill512', 128, 32, 512, 1, 0, 0),
    ('accepted', 128, 32, 1024, 0, 0, 1),
    ('byte-accepted', 128, 32, 1024, 1, 0, 1),
]


def main():
    rows = []
    for name in ['qwen38-bench8', 'ling-bench8', 'qwen38-long-de']:
        path = TRACES / (name + '.jsonl')
        contexts = {}
        for line in path.open():
            b = json.loads(line)
            if 'event' not in b:
                contexts[b['context']] = b['draft']
        target = next(k for k, draft in contexts.items() if not draft)
        for policy, decay, admit, prefill, byte, soft, accepted in POLICIES:
            warm = name == 'qwen38-bench8'
            env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_MOE_')}
            env.update(GGML_MOE_HYBRID_BYTE_VALUE=str(byte), GGML_MOE_HYBRID_LAYER_SOFT_PCT=str(soft),
                       GGML_MOE_REPLAY_ACCEPTED_ONLY=str(accepted), GGML_MOE_REPLAY_WARM=str(int(warm)))
            cmd = [str(ROOT / 'build-cpu/bin/test-moe-cache'), '--replay-log', str(path), '0', str(admit),
                   str(prefill), '0', str(decay), PROFILE if warm else '-', str(target)]
            run = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (OUT / f'replay-{name}-{policy}.log').write_text(run.stdout)
            row = dict(trace=name, policy=policy, command=cmd, env={k: v for k, v in env.items() if k.startswith('GGML_MOE_')}, exit=run.returncode)
            if run.returncode == 0:
                row.update(re.findall(r'([\w/]+)=([^\s]+)', next(l for l in run.stdout.splitlines() if l.startswith('context='))))
            rows.append(row)
            print(name, policy, row.get('decode_byte_hit_rate'), row.get('MiB/token'), run.returncode, flush=True)
    (OUT / 'replay.json').write_text(json.dumps(rows, indent=2) + '\n')
    assert all(r['exit'] == 0 and r['unresolved_spans'] == '0' for r in rows)


if __name__ == '__main__':
    main()
