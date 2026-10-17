#!/usr/bin/env python3
"""Run one quality check with a gateway lease and foreign-process CPU gating."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bench', ROOT / 'task-logs/round9/bench.py')
assert spec and spec.loader
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
stage = bench.stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen', 'ling'])
    parser.add_argument('policy', choices=['off', 'off-knobs', 'r7', 'policy', 'slots'])
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    name = f'{args.model}-{args.policy}-' + ('probe' if args.probe else 'ppl')
    out = ROOT / 'task-logs/round9' / name
    out.mkdir(exist_ok=True)
    cfg = stage.CONFIGS['q-warm-draft' if args.model == 'qwen' else 'ling-cache-mtp3']
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_MOE_')}
    env.update(cfg['env'])
    if args.policy in ['policy', 'off-knobs', 'slots']:
        env.update(GGML_MOE_HYBRID_BYTE_VALUE='1', GGML_MOE_HYBRID_DECAY='64')
    if args.policy == 'off-knobs':
        env.update(GGML_MOE_HYBRID_LAYER_SOFT_PCT='100', GGML_MOE_HYBRID_ADMIT_PREFILL='512')
    if args.policy.startswith('off'):
        env['GGML_MOE_HYBRID'] = '0'
    if args.policy == 'slots':
        env['GGML_MOE_HYBRID_MIB'] = '2944' if args.model == 'qwen' else '5376'
    model = cfg['cmd'][cfg['cmd'].index('-m') + 1]
    placement = ['-ncmoe', '48'] if args.model == 'qwen' else ['-cmoe']
    flags = ['-m', model, '-c', '512', '-b', '512', '-ub', '4', '-lm', 'none', '-fa', 'on', '-ngl', '99', '-t', '24'] + placement
    if args.probe:
        cmd = [str(ROOT / 'task-logs/round9/moe-probe'), str(out / 'logits')] + flags + ['-n', '32']
        env.update(GGML_MOE_PROBE_LOGITS='1', GGML_MOE_PROBE_WINDOW='4', GGML_MOE_PROBE_PREFIX='32')
    else:
        cmd = [str(ROOT / 'build-cuda/bin/llama-perplexity')] + flags + ['-f', '/opt/llm-gateway/data/build/texts/test-de.txt', '--chunks', '12']
    (out / 'command.json').write_text(json.dumps(dict(command=cmd, env={k: v for k, v in env.items() if k.startswith(('GGML_', 'LLAMA_', 'OMP_'))}), indent=2) + '\n')
    run = bench.Run(out, min(time.time() + 1800, 1792277760))
    for attempt in range(1, 4):
        run.quiet()
        try:
            stage.gw('POST', '/api/v1/gpu/free', {})
            run.held = True
            run.renew(force=True)
            if any(mib > 300 for mib in run.gpu_apps().values()):
                raise RuntimeError('foreign GPU allocation above 300 MiB')
            run.previous_gpu_pids = set(run.gpu_apps())
            with (out / f'attempt-{attempt}.log').open('w') as output:
                run.child = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                pid = run.child.pid
                busy = stage.busy_jiffies()
                own = stage.proc_jiffies([pid])
                t0 = last = time.monotonic()
                foreign_ticks = 0
                peak_vram = 0
                while run.child.poll() is None:
                    time.sleep(1)
                    now = time.monotonic()
                    busy_next = stage.busy_jiffies()
                    own_next = stage.proc_jiffies([pid])
                    if own_next >= own:
                        foreign_ticks += max(0, busy_next - busy - own_next + own)
                    busy, own = busy_next, own_next
                    run.check_gpu()
                    peak_vram = max(peak_vram, sum(run.gpu_apps().values()))
                    run.renew()
                    if now > run.end:
                        raise TimeoutError('quality deadline')
                    last = now
                code = run.child.returncode
                foreign = foreign_ticks / stage.HZ / (last - t0)
        finally:
            run.stop()
        accepted = code == 0 and foreign < 1.0
        run.event(kind='quality', name=name, attempt=attempt, exit=code, foreign_cores=foreign,
                  wall_s=last - t0, peak_vram_mib=peak_vram, accepted=accepted)
        if code:
            return code
        if accepted:
            (out / 'accepted.txt').write_text(f'attempt-{attempt}.log\n')
            return 0
    return 75


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
