#!/usr/bin/env python3
"""Bounded Nsight diagnosis of the real cached four-token Qwen path."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import bench

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profiler', choices=['nsys', 'ncu'])
    args = parser.parse_args()
    out = ROOT / 'task-logs/round9' / ('gpu-' + args.profiler)
    out.mkdir(exist_ok=True)
    cfg = bench.stage.CONFIGS['q-warm-draft']
    model = cfg['cmd'][cfg['cmd'].index('-m') + 1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_MOE_', 'LLAMA_ARG_SPEC_'))}
    settings = {**cfg['env'], 'GGML_CUDA_DISABLE_GRAPHS': '1',
                'GGML_MOE_PROBE_WINDOW': '4', 'GGML_MOE_PROBE_PREFIX': '4',
                'NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT': '1',
                'XDG_CONFIG_HOME': str(out / 'config'), 'XDG_CACHE_HOME': str(out / 'cache')}
    env.update(settings)
    probe = [str(ROOT / 'task-logs/round9/moe-probe'), str(out / 'probe'), '-m', model,
             '-c', '512', '-b', '512', '-ub', '4', '-n', '16', '-lm', 'none',
             '-fa', 'on', '-ngl', '99', '-ncmoe', '48', '-t', '24']
    if args.profiler == 'nsys':
        cmd = ['timeout', '--signal=TERM', '--kill-after=20', '600', 'nsys', 'profile',
               '--trace=cuda,osrt', '--sample=none', '--cpuctxsw=none', '--export=sqlite',
               '--force-overwrite=true', '-o', str(out / 'trace')] + probe
    else:
        cmd = ['sudo', '-n', 'env', 'NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT=1',
               'timeout', '--signal=TERM', '--kill-after=20', '600', 'ncu',
               '--set', 'full', '--kernel-name-base', 'demangled',
               '--kernel-name', 'regex:mul_mat_vec_q_moe.*true', '--launch-skip', '144',
               '--launch-count', '6', '--force-overwrite', '--export', str(out / 'kernels'),
               'env'] + [f'{key}={value}' for key, value in settings.items()] + probe
    (out / 'command.json').write_text(json.dumps(dict(command=cmd, env=settings,
        graphs=False, performance_comparison=False, target_only=True,
        description='Four fixed topics; four-token target windows; no drafter; graphs disabled for Nsight 2022.4.'), indent=2) + '\n')
    run = bench.Run(out, min(time.time() + 900, 1792277760))
    run.quiet()
    code = -1
    try:
        bench.stage.gw('POST', '/api/v1/gpu/free', {})
        run.held = True
        run.renew(force=True)
        if any(mib > 300 for mib in run.gpu_apps().values()):
            raise RuntimeError('foreign GPU allocation')
        run.previous_gpu_pids = set(run.gpu_apps())
        with (out / 'run.log').open('w') as stream:
            run.child = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        run.event(kind='profiler_start', profiler=args.profiler, pid=run.child.pid)
        while run.child.poll() is None:
            time.sleep(2)
            run.check_gpu()
            run.renew()
            if time.time() > run.end:
                raise TimeoutError('profile deadline')
        code = run.child.returncode
    finally:
        if run.child is not None and run.child.poll() is None and args.profiler == 'ncu':
            # Only the process group created by this harness; root owns ncu's children.
            subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', '-' + str(run.child.pid)], check=False)
            run.child.wait(timeout=30)
        run.stop()
    run.event(kind='profiler_complete', exit=code, performance_comparison=False)
    if code == 0 and args.profiler == 'ncu':
        with (out / 'metrics.csv').open('w') as stream:
            subprocess.run(['ncu', '--import', str(out / 'kernels.ncu-rep'), '--page', 'raw', '--csv'],
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return code


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
