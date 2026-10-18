#!/usr/bin/env python3
"""Existing server, NVTX request ranges and one real prompt; no instrumentation build."""
import argparse
import ctypes
import functools
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import bench


@functools.lru_cache(maxsize=4)
def cpu_layout(pid):
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
    return bases[0], symbols


def cpu_counters(pid):
    base, symbols = cpu_layout(pid)
    with open(f'/proc/{pid}/mem', 'rb', buffering=0) as memory:
        result = {name: list(struct.unpack('<10Q', os.pread(memory.fileno(), 80, base + offset)))
                  for name, offset in symbols.items()}
    return dict(calls=result['_ZL5calls'], cpu_ns=result['_ZL6cpu_ns'])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('prompt')
    parser.add_argument('--graphs', action='store_true')
    parser.add_argument('--plain', action='store_true')
    args = parser.parse_args()
    injection = os.environ.get('LD_PRELOAD', '')
    bench.configure()
    if injection:
        os.environ['LD_PRELOAD'] = injection
    out = args.out.resolve()
    base = bench.stage.CONFIGS['q-warm-draft']
    cmd = list(base['cmd'])
    env = dict(base['env'])
    if not args.graphs:
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    if not args.plain:
        env['LD_PRELOAD'] = ':'.join(filter(None, [str(bench.ROOT / 'task-logs/round9/cpu-profile.so'), injection]))
        cmd += ['-lv', '5']
    env.update(XDG_CONFIG_HOME=str(out / 'config'), XDG_CACHE_HOME=str(out / 'cache'), CUDA_CACHE_PATH=str(out / 'cuda-cache'))
    cfg = dict(cmd=cmd, env=env)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')

    class OwnedRun(bench.Run):
        def renew(self, force=False):
            pass  # The outer profiler owns and renews the lease.

        tracing = False
        first_cpu = None

        def request(self, *request_args, **kwargs):
            original = bench.stage.urllib.request.urlopen
            owner = self
            seen = False

            class Response:
                def __init__(self, response):
                    self.response = response

                def __enter__(self):
                    self.response.__enter__()
                    return self

                def __exit__(self, *exc):
                    return self.response.__exit__(*exc)

                def __iter__(self):
                    nonlocal seen
                    for raw in self.response:
                        if owner.tracing and not seen and raw.startswith(b'data:') and b'[DONE]' not in raw:
                            data = json.loads(raw[5:])
                            if any((c.get('delta') or {}).get('content') or (c.get('delta') or {}).get('reasoning_content') for c in data.get('choices', [])):
                                seen = True
                                nvtx.nvtxMarkA(b'first_token')
                                if not args.plain:
                                    owner.first_cpu = cpu_counters(owner.child.pid)
                        yield raw

            bench.stage.urllib.request.urlopen = lambda *a, **kw: Response(original(*a, **kw))
            try:
                return super().request(*request_args, **kwargs)
            finally:
                bench.stage.urllib.request.urlopen = original

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
        run.tracing = True
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
            assert run.first_cpu is not None, 'no first-token boundary'
            decode = {key: [a-b for a,b in zip(cpu_after[key], run.first_cpu[key])] for key in cpu_before}
            assert all(n >= 0 for values in counters.values() for n in values)
            (out / 'cpu-counters.json').write_text(json.dumps(dict(before=cpu_before, first_token=run.first_cpu, after=cpu_after, delta=counters, decode_delta=decode,
                scope='Complete request and client first-token-to-end interval; boundary can lag server emission; atomic snapshots may straddle a CPU call; CPU graph elapsed time can overlap GPU work; buckets 1..8 tokens, 9 larger, 0 no expert op.'), indent=2) + '\n')
        run.event(kind='profile_request', prompt=args.prompt, graphs=args.graphs, plain=args.plain,
            waited_s=waited, pre_busy_cores=pre, foreign_cores=foreign, accepted=foreign < 1,
            wall_s=wall, **result)
    finally:
        run.stop()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
