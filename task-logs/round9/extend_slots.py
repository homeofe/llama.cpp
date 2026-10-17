#!/usr/bin/env python3
"""Use the measured free VRAM before advancing to the draft experiment."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def command(args, log, required=True):
    with (OUT / log).open('w') as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(log, result.returncode, flush=True)
    if required and result.returncode:
        raise RuntimeError(log)


def main():
    command(['python3', '-S', 'task-logs/round9/gpu_profile.py', 'ncu'], 'ncu-retry.log', False)
    command(['python3', '-S', 'task-logs/round9/fit.py', 'qwen', '3456'], 'qwen-fit-3456.log')
    rows = [json.loads(line) for line in (OUT / 'qwen-fit-3456/runs.jsonl').read_text().splitlines()]
    fit = next(row for row in rows if row['kind'] == 'fit')
    if fit['fits']:
        configs = json.loads((OUT / 'extra-configs.json').read_text())
        configs['qwen-r9-slots-3456'] = json.loads((OUT / 'qwen-fit-3456/config.json').read_text())
        (OUT / 'extra-configs.json').write_text(json.dumps(configs, indent=2) + '\n')
        command(['python3', '-u', 'task-logs/round9/bench.py',
                 '/opt/llm-gateway/data/moe-quant/qwen38/round9-slots-extra', '21:45',
                 '--rounds', '1', '--only', 'qwen-r9-slots-2944', 'qwen-r9-slots-3456'], 'bench-slots-extra.log')
        command(['python3', '-S', 'task-logs/round9/summarize.py',
                 '/opt/llm-gateway/data/moe-quant/qwen38/round9-slots-extra/runs.jsonl'], 'slots-extra-summary.log')
    for model, budget in [('qwen', 3456 if fit['fits'] else 2944), ('ling', 6144)]:
        command(['python3', '-S', 'task-logs/round9/quality.py', model, 'slots',
                 '--budget-mib', str(budget), '--tag', 'compaction0'], f'{model}-slots-quality.log')
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%H:%M:%S') + ' Runde 9: Erweiterte Cache-Budgets gemessen und auf Perplexitaet geprueft; DSpark folgt.\n')


if __name__ == '__main__':
    main()
