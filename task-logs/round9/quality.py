#!/usr/bin/env python3
"""Run one quality check with a gateway lease and foreign-process CPU gating."""
import argparse
import hashlib
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
    parser.add_argument('--budget-mib', type=int)
    parser.add_argument('--disable-thp', action='store_true')
    parser.add_argument('--tag', default='')
    parser.add_argument('--cpu-library', type=Path)
    parser.add_argument('--attempts', type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.cpu_library:
        args.cpu_library = args.cpu_library.resolve()
        if not args.tag or not args.cpu_library.is_relative_to(ROOT) or not args.cpu_library.is_file():
            parser.error('--cpu-library requires an existing worktree library and a distinct --tag')
    if args.budget_mib is not None and args.budget_mib <= 0:
        parser.error('--budget-mib must be positive')
    budget_tag = f'-{args.budget_mib}' if args.budget_mib is not None else ''
    name = f'{args.model}-{args.policy}{budget_tag}-' + ('probe' if args.probe else 'ppl')
    if args.disable_thp:
        name = name.replace('-ppl', '-nothp-ppl').replace('-probe', '-nothp-probe')
    if args.tag:
        if not args.tag.isalnum():
            parser.error('--tag must be alphanumeric')
        name = name.replace('-ppl', f'-{args.tag}-ppl').replace('-probe', f'-{args.tag}-probe')
    out = ROOT / 'task-logs/round9' / name
    out.mkdir(exist_ok=True)
    cfg = stage.CONFIGS['q-warm-draft' if args.model == 'qwen' else 'ling-cache-mtp3']
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_MOE_')}
    env.update(cfg['env'])
    library_hash = None
    if args.cpu_library:
        env.update(LD_PRELOAD=str(args.cpu_library), LD_LIBRARY_PATH=str(ROOT / 'build-cuda/bin'))
        library_hash = hashlib.sha256(args.cpu_library.read_bytes()).hexdigest()
    if args.policy in ['policy', 'off-knobs']:
        env.update(GGML_MOE_HYBRID_BYTE_VALUE='1', GGML_MOE_HYBRID_DECAY='64')
    if args.policy == 'off-knobs':
        env.update(GGML_MOE_HYBRID_LAYER_SOFT_PCT='100', GGML_MOE_HYBRID_ADMIT_PREFILL='512')
    if args.policy.startswith('off'):
        env['GGML_MOE_HYBRID'] = '0'
    if args.policy == 'slots':
        env['GGML_MOE_HYBRID_MIB'] = '2944' if args.model == 'qwen' else '5376'
    if args.budget_mib is not None:
        env['GGML_MOE_HYBRID_MIB'] = str(args.budget_mib)
    model = cfg['cmd'][cfg['cmd'].index('-m') + 1]
    placement = ['-ncmoe', '48'] if args.model == 'qwen' else ['-cmoe']
    flags = ['-m', model, '-c', '512', '-b', '512', '-ub', '4', '-lm', 'none', '-fa', 'on', '-ngl', '99', '-t', '24'] + placement
    if args.cpu_library:
        flags += ['-tb', '24', '--cpu-mask', 'ffffff', '--cpu-strict', '1',
                  '--cpu-mask-batch', 'ffffff', '--cpu-strict-batch', '1', '--poll', '1', '--poll-batch', '1']
    if args.probe:
        cmd = [str(ROOT / 'task-logs/round9/moe-probe'), str(out / 'logits')] + flags + ['-n', '32']
        env.update(GGML_MOE_PROBE_LOGITS='1', GGML_MOE_PROBE_WINDOW='4', GGML_MOE_PROBE_PREFIX='32')
    else:
        cmd = [str(ROOT / 'build-cuda/bin/llama-perplexity')] + flags + ['-f', '/opt/llm-gateway/data/build/texts/test-de.txt', '--chunks', '12']
    if args.disable_thp:
        cmd = [str(ROOT / 'task-logs/round9/no-thp')] + cmd
    (out / 'command.json').write_text(json.dumps(dict(command=cmd, cpu_library_sha256=library_hash,
        env={k: v for k, v in env.items() if k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'LD_'))}), indent=2) + '\n')
    run = bench.Run(out, min(time.time() + 1800, 1792277760))
    for attempt in range(1, args.attempts + 1):
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
                evaluation_ticks = 0
                evaluation_s = 0.0
                evaluation_started = False
                evaluation_finished = False
                peak_vram = 0
                while run.child.poll() is None:
                    time.sleep(1)
                    now = time.monotonic()
                    busy_next = stage.busy_jiffies()
                    own_next = stage.proc_jiffies([pid])
                    if own_next >= own:
                        # Keep signed deltas until the entire observation is summed.
                        # Independent /proc accounting updates can briefly disagree.
                        interval_foreign = busy_next - busy - own_next + own
                        foreign_ticks += interval_foreign
                        if evaluation_started and not evaluation_finished:
                            evaluation_ticks += interval_foreign
                            evaluation_s += now - last
                    busy, own = busy_next, own_next
                    log = (out / f'attempt-{attempt}.log').read_text()
                    ready = 'llama threadpool init' if args.probe else 'calculating perplexity over'
                    done = 'TOPIC agent' if args.probe else 'Final estimate: PPL ='
                    if ready in log and not evaluation_started:
                        if args.cpu_library:
                            libraries = sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                                                if '/libggml' in line or '/libllama' in line})
                            (out / 'mapped-libraries.json').write_text(json.dumps(libraries, indent=2) + '\n')
                            assert str(args.cpu_library) in libraries, 'requested CPU library was not loaded'
                            assert all('/build-cuda/bin/' in p or p == str(args.cpu_library) for p in libraries), 'unexpected library replacement'
                        evaluation_started = True
                    if done in log:
                        evaluation_finished = True
                    run.check_gpu()
                    peak_vram = max(peak_vram, sum(run.gpu_apps().values()))
                    run.renew()
                    if time.time() > run.end:
                        raise TimeoutError('quality deadline')
                    last = now
                code = run.child.returncode
                total_foreign = max(0, foreign_ticks) / stage.HZ / (last - t0)
                foreign = max(0, evaluation_ticks) / stage.HZ / evaluation_s if evaluation_s > 0 else None
        finally:
            run.stop()
        accepted = code == 0 and foreign is not None and foreign < 1.0
        run.event(kind='quality', name=name, attempt=attempt, exit=code, foreign_cores=foreign,
                  whole_process_foreign_cores=total_foreign, evaluation_wall_s=evaluation_s,
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
