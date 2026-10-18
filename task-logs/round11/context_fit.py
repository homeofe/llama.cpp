#!/usr/bin/env python3
"""Measure 32K fixed allocations before selecting the largest fitting cache."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

import bench

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'


def fit(name, cfg, stress):
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    run = bench.Run(directory, min(time.time() + 900, 1792334640))
    memory = []
    finished = threading.Event()

    def sample_memory():
        while not finished.is_set():
            memory.append(int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()))
            finished.wait(1)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    try:
        run.quiet()
        run.start(name, cfg)
        run.previous_gpu_pids = set(run.gpu_apps()) - {run.child.pid}
        run.gpu_pid = run.child.pid
        run.check_gpu()
        sampler.start()
        for label, text in [('warmup', 'Say hello in five words.'), ('long-context', stress)]:
            for attempt in range(1, 4):
                run.quiet()
                busy, own, start = bench.stage.busy_jiffies(), bench.stage.proc_jiffies([run.child.pid]), time.monotonic()
                response = run.request(name, cfg, text, 16)
                wall = time.monotonic() - start
                foreign = max(0, bench.stage.busy_jiffies() - busy - bench.stage.proc_jiffies([run.child.pid]) + own) / bench.stage.HZ / wall
                run.event(kind='fit_request', label=label, attempt=attempt, foreign_cores=foreign,
                    accepted=foreign < 1, prompt_chars=len(text), prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), response=response)
                if foreign < 1:
                    break
            else:
                raise RuntimeError('fit request did not pass the CPU gate')
        memory.append(int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()))
        run.check_gpu()
        log = (directory / (name + '.server.log')).read_text(errors='replace')
        assert 'n_ctx_slot = 32768' in log, 'context was silently reduced'
        peak = max(memory)
        run.event(kind='fit', vram_mib=peak, fits=peak <= 10700, vram_samples_mib=memory)
        return peak
    finally:
        finished.set()
        if sampler.is_alive():
            sampler.join(timeout=5)
        run.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--async-upload', action='store_true')
    args = parser.parse_args()
    bench.configure()
    control = 'ling-async' if args.async_upload else 'ling-control'
    base = copy.deepcopy(bench.stage.CONFIGS[control])
    base['cmd'][base['cmd'].index('-c') + 1] = '32768'
    base['cmd'] += ['-fit', 'off', '-lv', '4']
    off = copy.deepcopy(base)
    off['env']['GGML_MOE_HYBRID'] = '0'
    responses = [json.loads(line) for line in (ROOT / 'task-logs/round10/panels/responses.jsonl').read_text().splitlines()]
    stress = max((r['request']['messages'][0]['content'] for r in responses
        if r['request'].get('messages') and isinstance(r['request']['messages'][0].get('content'), str)), key=len)
    assert len(stress) > 50000, 'long-combine fixture missing'
    fixed = fit('ling-32k-off-fit', off, stress)
    budget = min(4608, ((10700 - fixed) // 64) * 64)
    if budget < 512:
        raise RuntimeError('not enough room for a useful cache')
    attempts = []
    for attempt in range(1, 4):
        base['env']['GGML_MOE_HYBRID_MIB'] = str(budget)
        memory = fit('ling-32k-cache-fit-' + str(budget), base, stress)
        attempts.append(dict(cache_budget_mib=budget, peak_vram_mib=memory))
        if memory > 10700:
            budget -= ((memory - 10700 + 63) // 64) * 64
        elif memory <= 10636 and budget < 4608:
            larger = min(4608, budget + ((10700 - memory) // 64) * 64)
            if any(r['cache_budget_mib'] == larger for r in attempts):
                break
            budget = larger
        else:
            break
    else:
        raise RuntimeError('cache fit did not converge in three measurements')
    assert memory <= 10700 and int(base['env']['GGML_MOE_HYBRID_MIB']) == budget
    configs = json.loads((OUT / 'configs.json').read_text())
    log_flag = base['cmd'].index('-lv')
    del base['cmd'][log_flag:log_flag + 2]
    configs['ling-32k'] = base
    configs['ling-8k'] = copy.deepcopy(bench.stage.CONFIGS[control])
    configs['ling-8k']['cmd'] += ['-fit', 'off']
    (OUT / 'configs.json').write_text(json.dumps(configs, indent=2) + '\n')
    (OUT / 'context-fit.json').write_text(json.dumps(dict(fixed_vram_mib=fixed, cache_budget_mib=budget,
        measured_vram_mib=memory, original_cache_budget_mib=4608, budget_step_mib=64, total_vram_ceiling_mib=10700,
        attempts=attempts, stress_prompt_chars=len(stress), stress_prompt_sha256=hashlib.sha256(stress.encode()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
