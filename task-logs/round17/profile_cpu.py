#!/usr/bin/env python3
"""Sample the aligned engine's CPU execution; all request rates are diagnostic."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from run_variants import GraphRun, configs, bench, HERE

out = HERE / 'cpu-aligned'
out.mkdir(exist_ok=False)
run = GraphRun(out, time.time() + 10 * 60)
cfg = configs()['strata-reference']
cfg['env'].update(GGML_CUDA_GRAPH_VARIANTS='1', GGML_CUDA_GRAPH_TRACE='0')
(out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
profiler = None
try:
    run.quiet()
    run.start('aligned', cfg)
    run.request('Say hello in five words.', 'warmup', 16)
    run.request(bench.PROMPTS[0][1], 'code-prime', 256)
    run.quiet()
    tids = sorted(Path(f'/proc/{run.child.pid}/task').iterdir())
    affinity = {}
    for tid in tids:
        affinity[tid.name] = [line for line in (tid / 'status').read_text().splitlines()
                              if line.startswith(('Name:', 'Cpus_allowed_list:', 'Mems_allowed_list:'))]
    (out / 'thread-affinity.json').write_text(json.dumps(affinity, indent=2) + '\n')
    cmd = ['sudo', '-n', 'perf', 'record', '--no-buildid', '-F', '199', '--call-graph', 'dwarf,4096',
           '-p', str(run.child.pid), '-o', str(out / 'perf.data'), '--', 'sleep', '25']
    (out / 'perf-command.json').write_text(json.dumps(cmd) + '\n')
    with (out / 'perf-record.txt').open('w') as log:
        profiler = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1)
        if profiler.poll() is not None:
            raise RuntimeError('perf exited before the requests')
        for tag, prompt in [('code-repeat', bench.PROMPTS[0][1]), ('english', bench.PROMPTS[4][1])]:
            run.event(kind='request', tag=tag, benchmark_valid=False, **run.request(prompt, tag))
        status = profiler.wait(timeout=35)
        if status:
            raise RuntimeError(f'perf exited {status}')
    run.event(kind='profile_complete', benchmark_valid=False)
finally:
    if profiler is not None and profiler.poll() is None:
        # The recorder has its own 25-second lifetime, even if a request fails.
        profiler.wait(timeout=35)
    run.stop()
    run.event(kind='released', gpu=run.gpu_apps())

with (out / 'perf-report.txt').open('w') as report:
    subprocess.run(['sudo', '-n', 'perf', 'report', '--stdio', '--no-children', '-g', 'none',
                    '--sort', 'dso,symbol', '--percent-limit', '0.25', '-i', str(out / 'perf.data')],
                   stdout=report, stderr=subprocess.STDOUT, check=True, timeout=60)
