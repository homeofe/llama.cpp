#!/usr/bin/env python3
"""Fit Ling DSpark without cache, then with a conservative actual-VRAM budget."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import bench

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'
DRAFT = '/opt/llm-gateway/models/Ling-3.0-flash-GGUF/dspark/Ling-3.0-flash-DSpark-Q8_0.gguf'


def fit(name, cfg):
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    run = bench.Run(directory, min(time.time() + 900, 1792277760))
    try:
        run.quiet()
        run.start(name, cfg)
        run.check_gpu()
        response = run.request(name, cfg, 'Say hello in five words.', 16)
        run.check_gpu()
        memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                             '--format=csv,noheader,nounits'], text=True).strip())
        run.event(kind='fit', vram_mib=memory, fits=memory <= 10700, response=response)
        return memory
    finally:
        run.stop()


def main():
    for key in list(os.environ):
        if key.startswith('GGML_MOE_HYBRID') or key == 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH':
            del os.environ[key]
    cmd = list(bench.stage.CONFIGS['ling-mtp3']['cmd'])
    cmd[cmd.index('--spec-type') + 1] = 'draft-dspark'
    cmd[cmd.index('--spec-draft-n-max') + 1] = '7'
    cmd += ['-md', DRAFT, '-ngld', '99', '--spec-draft-n-ubatch', '32', '--spec-draft-p-min', '0']
    off = dict(cmd=cmd, env=bench.stage.BOLT)
    fixed = fit('ling-dspark-off-fit', off)
    budget = min(5120, ((10700 - fixed - 256) // 256) * 256)
    if budget < 1024:
        raise RuntimeError(f'insufficient DSpark cache headroom: fixed allocation {fixed} MiB')
    cached = dict(cmd=cmd, env={**bench.stage.CONFIGS['ling-cache-mtp3']['env'],
                               'GGML_MOE_HYBRID_MIB': str(budget)})
    memory = fit('ling-dspark-cache-fit', cached)
    if memory > 10700:
        raise RuntimeError(f'DSpark cache allocation exceeds limit: {memory} MiB')
    path = OUT / 'extra-configs.json'
    extra = json.loads(path.read_text()) if path.exists() else {}
    extra.update({'ling-dspark-off': off, 'ling-dspark-cache': cached})
    path.write_text(json.dumps(extra, indent=2) + '\n')
    with (OUT / 'dspark-budget-quality.log').open('w') as stream:
        subprocess.run(['python3', '-S', 'task-logs/round9/quality.py', 'ling', 'slots',
                        '--budget-mib', str(budget), '--tag', 'dspark'], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    quality = OUT / f'ling-slots-{budget}-dspark-ppl'
    log = (quality / (quality / 'accepted.txt').read_text().strip()).read_text()
    match = re.search(r'Final estimate: PPL = ([0-9.]+)', log)
    assert match, 'missing draft-cache-budget quality result'
    ppl = float(match[1])
    (OUT / 'dspark-budget-quality.json').write_text(json.dumps(dict(budget_mib=budget,
        cache_off_ppl=4.5559, candidate_ppl=ppl, passes=ppl <= 4.5559 * 1.005), indent=2) + '\n')
    assert ppl <= 4.5559 * 1.005, 'DSpark cache budget fails quality gate'
    print(json.dumps(dict(fixed_mib=fixed, budget_mib=budget, cached_mib=memory)))


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    main()
