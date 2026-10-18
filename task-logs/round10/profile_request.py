#!/usr/bin/env python3
"""Existing server, NVTX request ranges and one real prompt; no instrumentation build."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import bench


def cpu_counters(pid):
    library = (bench.ROOT / 'task-logs/round9/cpu-profile.so').resolve()
    symbols = {}
    for line in subprocess.check_output(['nm', '-a', str(library)], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] in ['_ZL5calls', '_ZL6cpu_ns']:
            symbols[fields[2]] = int(fields[0], 16)
    assert len(symbols) == 2
    maps = Path(f'/proc/{pid}/maps').read_text().splitlines()
    bases = [int(line.split()[0].split('-')[0], 16) for line in maps
             if line.split()[-1] == str(library) and int(line.split()[2], 16) == 0]
    assert len(bases) == 1
    with open(f'/proc/{pid}/mem', 'rb', buffering=0) as memory:
        result = {name: list(struct.unpack('<10Q', os.pread(memory.fileno(), 80, bases[0] + offset)))
                  for name, offset in symbols.items()}
    return dict(calls=result['_ZL5calls'], cpu_ns=result['_ZL6cpu_ns'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('prompt')
    parser.add_argument('--graphs', action='store_true')
    parser.add_argument('--plain', action='store_true')
    args = parser.parse_args()
    bench.configure()
    out = args.out.resolve()
    base = bench.stage.CONFIGS['q-warm-draft']
    cmd = list(base['cmd'])
    env = dict(base['env'])
    if not args.graphs:
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    if not args.plain:
        env['LD_PRELOAD'] = str(bench.ROOT / 'task-logs/round9/cpu-profile.so')
        cmd += ['-lv', '5']
    env.update(XDG_CONFIG_HOME=str(out / 'config'), XDG_CACHE_HOME=str(out / 'cache'), CUDA_CACHE_PATH=str(out / 'cuda-cache'))
    cfg = dict(cmd=cmd, env=env)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')

    class OwnedRun(bench.Run):
        def renew(self, force=False):
            pass  # The outer profiler owns and renews the lease.

        def start(self, name, config):
            import subprocess
            with (self.out / (name + '.server.log')).open('w') as stream:
                self.child = subprocess.Popen(config['cmd'], env={**os.environ, **config['env']},
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            (out / 'owned-pids.json').write_text(json.dumps(dict(wrapper=os.getpid(), server=self.child.pid)) + '\n')
            self.gpu_pid = self.child.pid
            self.previous_gpu_pids = set()
            for _ in range(450):
                if self.child.poll() is not None:
                    raise RuntimeError('server stopped before health')
                try:
                    with bench.stage.urllib.request.urlopen('http://127.0.0.1:8799/health', timeout=2) as response:
                        if response.status == 200:
                            return
                except Exception:
                    pass
                time.sleep(2)
            raise TimeoutError('server health')

    run = OwnedRun(out, min(time.time() + 1200, 1792320900))
    nvtx = ctypes.CDLL('libnvToolsExt.so.1')
    nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    nvtx.nvtxMarkA.argtypes = [ctypes.c_char_p]
    name = 'q-warm-draft'
    try:
        run.start(name, cfg)
        run.request(name, cfg, 'Say hello in five words.', 16)
        waited, pre = run.quiet()
        run.check_gpu()
        prompt = dict(bench.stage.PROMPTS)[args.prompt]
        cpu_before = cpu_counters(run.child.pid) if not args.plain else None
        nvtx.nvtxRangePushA(b'round10')
        nvtx.nvtxMarkA(('epoch_ns=' + str(time.time_ns()) + ';monotonic_ns=' + str(time.monotonic_ns())).encode())
        b0, o0, t0 = bench.stage.busy_jiffies(), bench.stage.proc_jiffies([run.child.pid]), time.monotonic()
        nvtx.nvtxRangePushA(args.prompt.encode())
        result = run.request(name, cfg, prompt, 256)
        nvtx.nvtxRangePop()
        wall = time.monotonic() - t0
        foreign = max(0, bench.stage.busy_jiffies() - b0 - bench.stage.proc_jiffies([run.child.pid]) + o0) / bench.stage.HZ / wall
        nvtx.nvtxMarkA(('epoch_ns=' + str(time.time_ns()) + ';monotonic_ns=' + str(time.monotonic_ns())).encode())
        nvtx.nvtxRangePop()
        if cpu_before is not None:
            cpu_after = cpu_counters(run.child.pid)
            counters = {key: [a-b for a,b in zip(cpu_after[key], cpu_before[key])] for key in cpu_before}
            assert all(n >= 0 for values in counters.values() for n in values)
            (out / 'cpu-counters.json').write_text(json.dumps(dict(before=cpu_before, after=cpu_after, delta=counters,
                scope='Complete request; CPU graph elapsed time can overlap GPU work; buckets 1..8 tokens, 9 larger, 0 no expert op.'), indent=2) + '\n')
        run.event(kind='profile_request', prompt=args.prompt, graphs=args.graphs, plain=args.plain,
            waited_s=waited, pre_busy_cores=pre, foreign_cores=foreign, accepted=foreign < 1,
            wall_s=wall, **result)
    finally:
        run.stop()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
