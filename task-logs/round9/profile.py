#!/usr/bin/env python3
"""Profile our server with bounded perf sampling and GPU telemetry."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import bench

ROOT = Path(__file__).resolve().parents[2]


class ProfileRun(bench.Run):
    def __init__(self, out, end, perf):
        super().__init__(out, end)
        self.perf_enabled = perf
        self.telemetry = []
        self.recorders = []
        self.counters = []

    def start(self, name, cfg):
        super().start(name, cfg)
        self.check_gpu()
        prefix = self.out / f'{name}-{self.child.pid}'
        status = Path(f'/proc/{self.child.pid}/status').read_text()
        prefix.with_suffix('.memory.txt').write_text('\n'.join(line for line in status.splitlines()
            if line.startswith(('VmRSS:', 'VmLck:', 'VmSwap:', 'Threads:', 'Cpus_allowed_list:', 'Mems_allowed_list:'))) + '\n'
            + Path(f'/proc/{self.child.pid}/smaps_rollup').read_text())
        threads = []
        for directory in Path(f'/proc/{self.child.pid}/task').iterdir():
            try:
                lines = (directory / 'status').read_text().splitlines()
                threads.append(dict(tid=int(directory.name), status=[line for line in lines
                    if line.startswith(('Name:', 'Cpus_allowed_list:'))]))
            except FileNotFoundError:
                pass
        prefix.with_suffix('.affinity.json').write_text(json.dumps(threads, indent=2) + '\n')
        fields = 'timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.current.sm,clocks.current.memory,power.draw,utilization.gpu,utilization.memory,memory.used'
        with prefix.with_suffix('.gpu.csv').open('w') as stream:
            self.telemetry.append(subprocess.Popen(['nvidia-smi', '--query-gpu=' + fields,
                                                   '--format=csv', '-lms', '200'], stdout=stream, stderr=subprocess.STDOUT))
        with prefix.with_suffix('.pcie.txt').open('w') as stream:
            self.telemetry.append(subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'pucvt', '-o', 'DT', '-d', '1', '-c', '120'],
                                                   stdout=stream, stderr=subprocess.STDOUT))
        if self.perf_enabled:
            command = ['sudo', '-n', 'perf', 'record', '-F', '499', '-e', 'cycles:u',
                       '-p', str(self.child.pid), '-o', '-', '--', 'sleep', '120']
            with prefix.with_suffix('.perf.data').open('wb') as data, prefix.with_suffix('.perf.log').open('w') as log:
                proc = subprocess.Popen(command, stdout=data, stderr=log)
            self.recorders.append((proc, prefix))
            self.event(kind='profiler', command=command, target_pid=self.child.pid, prefix=str(prefix))
            stats = ['sudo', '-n', 'perf', 'stat', '-x', ',', '-e', 'cycles:u,instructions:u,cache-misses:u,dTLB-load-misses:u',
                     '-p', str(self.child.pid), '--', 'sleep', '120']
            with prefix.with_suffix('.counters.csv').open('w') as log:
                proc = subprocess.Popen(stats, stdout=subprocess.DEVNULL, stderr=log)
            self.counters.append((proc, prefix))
            self.event(kind='hardware_counters', command=stats, prefix=str(prefix))

    def stop(self):
        try:
            super().stop()
        finally:
            for proc in self.telemetry:
                proc.terminate()
                proc.wait(timeout=5)
            self.telemetry = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=['q-warm-draft', 'ling-cache-mtp3'])
    parser.add_argument('--perf', action='store_true')
    args = parser.parse_args()
    for key in list(os.environ):
        if key.startswith('GGML_MOE_HYBRID') or key == 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH':
            del os.environ[key]
    out = ROOT / 'task-logs/round9' / f'{args.config}-profile'
    out.mkdir(exist_ok=True)
    base = bench.stage.CONFIGS[args.config]
    cfg = dict(cmd=base['cmd'], env={**base['env'], 'LD_PRELOAD': str(ROOT / 'task-logs/round9/cpu-profile.so')})
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    run = ProfileRun(out, min(time.time() + 1200, 1792277760), args.perf)
    try:
        run.config_round(args.config, cfg, 1)
    finally:
        run.stop()
    for proc, prefix in run.recorders:
        while proc.poll() is None:
            time.sleep(1)
        run.event(kind='profiler_complete', exit=proc.returncode, prefix=str(prefix))
        if proc.returncode == 0:
            with prefix.with_suffix('.perf.txt').open('w') as output, prefix.with_suffix('.perf.data').open('rb') as data:
                subprocess.run(['perf', 'report', '--stdio', '--no-children', '--sort', 'dso,symbol',
                                '-i', '-'], stdin=data, stdout=output, stderr=subprocess.STDOUT, check=True)
    for proc, prefix in run.counters:
        while proc.poll() is None:
            time.sleep(1)
        run.event(kind='hardware_counters_complete', exit=proc.returncode, prefix=str(prefix))
    run.event(kind='profile_complete', performance_comparison=False)


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    main()
