#!/usr/bin/env python3
"""Run bounded GPU diagnosis and one screening round after allocation probes."""
import json
from pathlib import Path
import subprocess
import time

import bench

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def command(args, log, required=True):
    with (OUT / log).open('w') as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(log, 'exit', result.returncode, flush=True)
    if result.returncode and required:
        raise RuntimeError(log)


def main():
    deadline = time.monotonic() + 2700
    while not (OUT / 'fit-complete.txt').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('allocation probes have not completed')
        time.sleep(5)
    for profiler in ['nsys', 'ncu']:
        command(['python3', '-S', 'task-logs/round9/gpu_profile.py', profiler], f'{profiler}-run.log', required=False)
    configs = {}
    names = []
    for model, budget, control in [('qwen', 2944, 'q-warm-draft'), ('ling', 6144, 'ling-cache-mtp3')]:
        directory = OUT / f'{model}-fit-{budget}'
        records = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
        fit = next(row for row in records if row['kind'] == 'fit')
        if not fit['fits']:
            print('skip unsafe allocation', model, fit['vram_mib'], flush=True)
            continue
        name = f'{model}-r9-slots-{budget}'
        configs[name] = json.loads((directory / 'config.json').read_text())
        names.extend([control, name])
    (OUT / 'extra-configs.json').write_text(json.dumps(configs, indent=2) + '\n')
    if names:
        command(['python3', '-u', 'task-logs/round9/bench.py',
                 '/opt/llm-gateway/data/moe-quant/qwen38/round9-slots', '21:30',
                 '--rounds', '1', '--only'] + names, 'bench-slots.log')
        command(['python3', '-S', 'task-logs/round9/summarize.py',
                 '/opt/llm-gateway/data/moe-quant/qwen38/round9-slots/runs.jsonl'], 'slots-summary.log')
    (OUT / 'slots-complete.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')


if __name__ == '__main__':
    main()
