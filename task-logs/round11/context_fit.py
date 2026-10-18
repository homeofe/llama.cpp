#!/usr/bin/env python3
"""Measure 32K fixed allocations before selecting the largest fitting cache."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import time

import bench

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'


def fit(name, cfg):
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    run = bench.Run(directory, min(time.time() + 900, 1792334640))
    try:
        run.quiet()
        run.start(name, cfg)
        run.previous_gpu_pids = set(run.gpu_apps()) - {run.child.pid}
        run.gpu_pid = run.child.pid
        run.check_gpu()
        response = run.request(name, cfg, 'Say hello in five words.', 16)
        memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        run.check_gpu()
        log = (directory / (name + '.server.log')).read_text(errors='replace')
        assert 'n_ctx_slot = 32768' in log, 'context was silently reduced'
        run.event(kind='fit', vram_mib=memory, fits=memory <= 10700, response=response)
        return memory
    finally:
        run.stop()


def main():
    bench.configure()
    base = copy.deepcopy(bench.stage.CONFIGS['ling-dspark-n3-p00'])
    base['cmd'][base['cmd'].index('-c') + 1] = '32768'
    base['cmd'] += ['-fit', 'off', '-lv', '4']
    off = copy.deepcopy(base)
    off['env']['GGML_MOE_HYBRID'] = '0'
    fixed = fit('ling-32k-off-fit', off)
    budget = min(4608, ((10700 - fixed) // 64) * 64)
    if budget < 512:
        raise RuntimeError('not enough room for a useful cache')
    base['env']['GGML_MOE_HYBRID_MIB'] = str(budget)
    memory = fit('ling-32k-cache-fit', base)
    if memory > 10700:
        raise RuntimeError('actual 32K allocation exceeds the declared VRAM ceiling')
    configs = json.loads((OUT / 'configs.json').read_text())
    configs['ling-32k'] = base
    configs['ling-8k'] = copy.deepcopy(bench.stage.CONFIGS['ling-dspark-n3-p00'])
    (OUT / 'configs.json').write_text(json.dumps(configs, indent=2) + '\n')
    (OUT / 'context-fit.json').write_text(json.dumps(dict(fixed_vram_mib=fixed, cache_budget_mib=budget,
        measured_vram_mib=memory, original_cache_budget_mib=4608, budget_step_mib=64, total_vram_ceiling_mib=10700), indent=2) + '\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
