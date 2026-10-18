#!/usr/bin/env python3
"""Existing server, NVTX request ranges and one real prompt; no instrumentation build."""
import argparse
import ctypes
import functools
import hashlib
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


def logger_epoch(pid, spawn_epoch_ns, out):
    library = (bench.ROOT / 'build-cuda/bin/libllama-common.so').resolve()
    symbols = subprocess.check_output(['nm', '-a', str(library)], text=True).splitlines()
    offset = next(int(line.split()[0], 16) for line in symbols if line.endswith(' _ZZ15common_log_mainvE3log'))
    disassembly = subprocess.check_output(['objdump', '-d', '--start-address=0x2e2670', '--stop-address=0x2e2750', str(library)], text=True)
    assert 'mov    %rdx,0xa0(%rbx)' in disassembly, 'logger layout changed; inspect before reading'
    (out / 'logger-layout.txt').write_text(disassembly)
    maps = Path(f'/proc/{pid}/maps').read_text().splitlines()
    base = next(int(line.split()[0].split('-')[0], 16) for line in maps
                if line.split()[-1] == str(library) and int(line.split()[2], 16) == 0)
    with open(f'/proc/{pid}/mem', 'rb', buffering=0) as memory:
        pointer, = struct.unpack('<Q', os.pread(memory.fileno(), 8, base + offset))
        epoch_us, = struct.unpack('<Q', os.pread(memory.fileno(), 8, pointer + 0xa0))
    assert spawn_epoch_ns - 1_000_000_000 < epoch_us * 1000 < time.time_ns()
    (out / 'logger-epoch.json').write_text(json.dumps(dict(epoch_us=epoch_us, field_offset=0xa0,
        library=str(library), sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        method='Read-only parent /proc/PID/mem; layout checked against this existing binary disassembly.'), indent=2) + '\n')



def owned_processes(root_pid):
    parents = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                parents[int(path.name)] = int((path / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    owned = {root_pid}
    while True:
        found = {pid for pid, parent in parents.items() if parent in owned}
        if found <= owned:
            return sorted(owned)
        owned |= found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('prompt')
    parser.add_argument('--graphs', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plain', action='store_true')
    mode.add_argument('--counters-only', action='store_true')
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
    cmd += ['-lv', '4' if args.plain else '5', '--log-prefix', '--log-timestamps', '--log-colors', 'off']
    env.update(XDG_CONFIG_HOME=str(out / 'config'), XDG_CACHE_HOME=str(out / 'cache'), CUDA_CACHE_PATH=str(out / 'cuda-cache'))
    cfg = dict(cmd=cmd, env=env)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')

    class OwnedRun(bench.Run):
        def renew(self, force=False):
            pass  # The outer profiler owns and renews the lease.

        tracing = False
        first_cpu = None
        first_epoch_ns = None

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
                                owner.first_epoch_ns = time.time_ns()
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
            self.spawn_epoch_ns = time.time_ns()
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
        logger_epoch(run.child.pid, run.spawn_epoch_ns, out)
        (out / 'server.maps.txt').write_text(Path(f'/proc/{run.child.pid}/maps').read_text())
        affinity = {}
        for status in Path(f'/proc/{run.child.pid}/task').glob('*/status'):
            try:
                fields = dict(line.split(':', 1) for line in status.read_text().splitlines() if ':' in line)
                affinity[status.parent.name] = {key: fields[key].strip() for key in ['Name', 'Cpus_allowed_list']}
            except FileNotFoundError:
                pass
        (out / 'server-affinity.json').write_text(json.dumps(affinity, indent=2) + '\n')
        waited, pre = run.quiet()
        run.check_gpu()
        prompt = dict(bench.stage.PROMPTS)[args.prompt]
        cpu_before = cpu_counters(run.child.pid) if not args.plain else None
        nvtx.nvtxRangePushA(b'round10')
        nvtx.nvtxMarkA(('epoch_ns=' + str(time.time_ns()) + ';monotonic_ns=' + str(time.monotonic_ns())).encode())
        profiler_root = json.loads((out / 'profiler-owner.json').read_text())['pid']
        own_pids = owned_processes(profiler_root)
        assert run.child.pid in own_pids
        b0, o0, t0 = bench.stage.busy_jiffies(), bench.stage.proc_jiffies(own_pids), time.monotonic()
        run.tracing = True
        nvtx.nvtxRangePushA(args.prompt.encode())
        request_start_epoch_ns = time.time_ns()
        result = run.request(name, cfg, prompt, 256)
        request_end_epoch_ns = time.time_ns()
        nvtx.nvtxRangePop()
        wall = time.monotonic() - t0
        foreign = max(0, bench.stage.busy_jiffies() - b0 - bench.stage.proc_jiffies(own_pids) + o0) / bench.stage.HZ / wall
        nvtx.nvtxMarkA(('epoch_ns=' + str(time.time_ns()) + ';monotonic_ns=' + str(time.monotonic_ns())).encode())
        nvtx.nvtxRangePop()
        (out / 'request-boundaries.json').write_text(json.dumps(dict(start_epoch_ns=request_start_epoch_ns,
            first_token_epoch_ns=run.first_epoch_ns, end_epoch_ns=request_end_epoch_ns,
            scope='Client observation boundaries; first-token delivery can lag server emission.'), indent=2) + '\n')
        if cpu_before is not None:
            cpu_after = cpu_counters(run.child.pid)
            counters = {key: [a-b for a,b in zip(cpu_after[key], cpu_before[key])] for key in cpu_before}
            assert run.first_cpu is not None, 'no first-token boundary'
            decode = {key: [a-b for a,b in zip(cpu_after[key], run.first_cpu[key])] for key in cpu_before}
            assert all(n >= 0 for values in counters.values() for n in values)
            (out / 'cpu-counters.json').write_text(json.dumps(dict(before=cpu_before, first_token=run.first_cpu, after=cpu_after, delta=counters, decode_delta=decode,
                scope='Complete request and client first-token-to-end interval; boundary can lag server emission; atomic snapshots may straddle a CPU call; CPU graph elapsed time can overlap GPU work; buckets 1..8 tokens, 9 larger, 0 no expert op.'), indent=2) + '\n')
        run.event(kind='profile_request', prompt=args.prompt, graphs=args.graphs, plain=args.plain, counters_only=args.counters_only,
            waited_s=waited, pre_busy_cores=pre, owned_profiler_processes=own_pids, foreign_cores=foreign, accepted=foreign < 1,
            wall_s=wall, **result)
    finally:
        run.stop()
        (out / 'server-stopped.json').write_text(json.dumps(dict(epoch_ns=time.time_ns())) + '\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
