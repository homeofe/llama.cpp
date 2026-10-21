#!/usr/bin/env python3
"""Round 14 phase 0: nsys profile of the FINAL stack (q-vibe-async) on the round-1 code
prompt, same methodology as round 1 (graphs disabled, one steady request, quiet gate, GPU
held), plus a VRAM sampler that records used/free while the server is loaded - the slots
bump candidate needs the real headroom number. Usage:
    python3 profile.py [prompt_index]     (default 0 = code-duration)
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path('/opt/llm-gateway/data/build/llama-vibe')
sys.path.insert(0, str(ROOT / 'task-logs/vibe-round1'))
sys.path.insert(0, '/opt/llm-gateway/data/moe-quant/qwen38')
import stage1  # noqa: E402
import stage_b  # noqa: E402

OUT = ROOT / 'task-logs/vibe-round14'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prompt_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = time.time() + 5400
    cfg = stage_b.CONFIGS['q-vibe-async']
    env = dict(os.environ, **cfg['env'], GGML_CUDA_DISABLE_GRAPHS='1')
    ours = str(ROOT / 'build-cuda/bin/llama-server')
    cmd = [(ours if c == stage_b.L5 else c) for c in cfg['cmd']]
    port = int(cfg['cmd'][cfg['cmd'].index('--port') + 1])
    log = stage1.Ledger(OUT)
    log(kind='start', command='round14-profile', config='q-vibe-async',
        prompt=stage_b.PROMPTS[prompt_idx][0], graphs='disabled')

    samples = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.free',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10)
                used, free = r.stdout.strip().split(',')
                samples.append((time.time(), int(used), int(free)))
            except Exception:
                pass
            time.sleep(5)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        stage1.run_nsys(OUT, log, 'async-' + stage_b.PROMPTS[prompt_idx][0], cmd, env,
                        '', stage_b.PROMPTS[prompt_idx][1], 'llama-server', end, port=port)
    finally:
        stop.set()
        t.join(timeout=15)
        if samples:
            # the load phase dominates the first minutes; report the tail (steady state)
            tail = samples[-12:]
            log(kind='vram_samples', n=len(samples),
                steady_used_mib=[s[1] for s in tail], steady_free_mib=[s[2] for s in tail],
                min_free_mib=min(s[2] for s in samples),
                min_free_after_load_mib=min(s[2] for s in samples[len(samples)//3:]))
    log(kind='complete', command='round14-profile')


if __name__ == '__main__':
    main()
