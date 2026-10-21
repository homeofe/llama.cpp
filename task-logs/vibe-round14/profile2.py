#!/usr/bin/env python3
"""Round 14 runner: quiet gate + GPU hold, then nsys with --capture-range=nvtx so the
report finalizes cleanly at the end of the profiled request (the round-10 recipe),
plus the VRAM sampler. Usage: python3 profile2.py [prompt_index]"""
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, '/opt/llm-gateway/data/build/llama-vibe/task-logs/vibe-round1')
import stage1  # noqa: E402

ROOT = Path('/opt/llm-gateway/data/build/llama-vibe')
OUT = ROOT / 'task-logs/vibe-round14'


def main():
    prompt_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = time.time() + 5400
    log = stage1.Ledger(OUT)
    log(kind='runner_start', capture='round14-capture-range', prompt_idx=prompt_idx)

    samples = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.free',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                used, free = r.stdout.strip().split(',')
                samples.append((time.time(), int(used), int(free)))
            except Exception:
                pass
            time.sleep(5)

    wt = threading.Thread(target=watch, daemon=True)
    wt.start()

    hold = stage1.Hold(log)
    proc = None
    try:
        waited = stage1.wait_quiet(end, None, timeout=1800)
        hold.acquire()
        log(kind='hold_acquired', waited_s=waited)
        for sub in ('tmp', 'config', 'cache'):
            (OUT / sub).mkdir(exist_ok=True)
        argv = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                '--capture-range=nvtx', '--nvtx-capture=round14', '--capture-range-end=stop',
                '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0',
                '--kill=none', '--force-overwrite=true', '--duration=1800', '--export=sqlite',
                '-o', str(OUT / 'trace'),
                'env', f'TMPDIR={OUT}/tmp', f'XDG_CONFIG_HOME={OUT}/config',
                f'XDG_CACHE_HOME={OUT}/cache',
                'python3', str(OUT / 'driver.py'), str(prompt_idx)]
        logf = (OUT / 'profile2.nsys.log').open('w')
        proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        logf.close()
        log(kind='nsys_pid', pid=proc.pid)
        deadline = time.time() + 3000
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(5)
        log(kind='nsys_exit', code=proc.poll())
    finally:
        hold.release()
        stop.set()
        wt.join(timeout=15)
        if samples:
            tail = samples[-12:]
            log(kind='vram_samples', n=len(samples),
                steady_used_mib=[s[1] for s in tail], steady_free_mib=[s[2] for s in tail],
                min_free_mib=min(s[2] for s in samples),
                min_free_after_load_mib=min(s[2] for s in samples[len(samples) // 3:]))
        log(kind='complete', sqlite=str(OUT / 'trace.sqlite'))


if __name__ == '__main__':
    main()
