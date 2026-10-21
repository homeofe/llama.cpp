#!/usr/bin/env python3
"""Two bounded diagnostic captures, never throughput benchmarks."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'round16'))
from native_reference import Reference, configs, bench, STRATA


def owned(root, out):
    parents, selected = {}, {root}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            parents[int(p.name)] = int((p / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            if (p / 'comm').read_text().strip() == 'nsys' and (
                'XDG_CONFIG_HOME=' + str(out / 'config')).encode() in (p / 'environ').read_bytes().split(b'\0'):
                selected.add(int(p.name))
        except (OSError, ValueError, IndexError):
            pass
    while True:
        more = {pid for pid, parent in parents.items() if parent in selected}
        if more <= selected:
            return sorted(selected)
        selected |= more


class Run(Reference):
    def renew(self, force=False):
        if self.held and (force or time.monotonic() - self.last_hold > 300):
            bench.gw('POST', '/api/v1/gpu/hold', {'reason': 'codex: bounded CPU GPU handoff diagnosis', 'ttlS': 900})
            self.last_hold = time.monotonic()

    def request(self, *args, **kwargs):
        def timeout(*_):
            raise TimeoutError('request deadline')
        old = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, min(90, max(1, self.end - time.time())))
        try:
            return super().request(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def start_bpf(pid, out):
    binary = STRATA / 'engine/strata'
    dispatch = '_ZN12_GLOBAL__N_116drive_pool_multiEPvPKfPKillPfl'
    phase = '_ZN6strata7kernels3cpu10ExpertPool9run_phaseEii'
    script = 'BEGIN { printf("READY\n"); }\n'
    for name, symbol, arguments in [('dispatch', dispatch, 'arg3, *(int64 *)(reg("sp") + 8)'),
                                     ('phase', phase, 'arg1, arg2')]:
        script += f'uprobe:{binary}:{symbol} /pid == {pid}/ {{ printf("B,{name},%llu,%d,%lld,%lld\\n", nsecs, tid, {arguments}); }}\n'
        script += f'uretprobe:{binary}:{symbol} /pid == {pid}/ {{ printf("E,{name},%llu,%d,0,0\\n", nsecs, tid); }}\n'
    # Keep newlines inside the BEGIN string escaped as bpftrace source.
    script = script.replace('printf("READY\n")', 'printf("READY\\n")')
    path = out / 'probes.bt'
    path.write_text(script)
    with (out / 'bpf.txt').open('w') as log:
        proc = subprocess.Popen(['sudo', '-n', 'bpftrace', '-q', '-B', 'line', '-p', str(pid), str(path)],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('bpftrace attachment failed; see bpf.txt')
            if 'READY' in (out / 'bpf.txt').read_text():
                return proc
            time.sleep(.1)
        raise TimeoutError('bpftrace attach deadline')
    except BaseException:
        stop_bpf(proc)
        raise


def stop_bpf(proc):
    if proc and proc.poll() is None:
        subprocess.run(['sudo', '-n', 'kill', '-INT', '--', '-' + str(proc.pid)], check=False)
        proc.wait(timeout=10)


def driver(a, out):
    run = Run(out, time.time() + a.minutes * 60)
    cfg = configs().get(a.engine)
    if cfg:
        preload = ':'.join(filter(None, [str(HERE / 'cpu_ranges.so'), os.environ.get('LD_PRELOAD', '')]))
        cfg['env'].update(LD_PRELOAD=preload, GGML_CUDA_GRAPH_VARIANTS='1', GGML_CUDA_GRAPH_TRACE='0')
    (out / 'llama-config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    nvtx = ctypes.CDLL('libnvToolsExt.so.1')
    nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    nvtx.nvtxMarkA.argtypes = [ctypes.c_char_p]
    tracer = None
    try:
        (out / 'owner.json').write_text(json.dumps({'pid': os.getpid()}) + '\n')
        run.quiet()
        run.start(a.engine, cfg)
        run.request('Say hello in five words.', 'warmup', 16)
        prompt = bench.PROMPTS[0][1]
        run.quiet()
        run.event(kind='warm_request', benchmark_valid=False, **run.request(prompt, 'novel', 256))
        if a.engine == 'strata':
            tracer = start_bpf(run.engine[0], out)
        run.quiet()
        pids = sorted(set(owned(os.getpid(), out)) | set(run.owned()))
        (out / 'owned.json').write_text(json.dumps(pids) + '\n')
        load = os.getloadavg()[0]
        b, o, t = bench.busy_jiffies(), bench.proc_jiffies(pids), time.monotonic()
        nvtx.nvtxRangePushA(b'handoff-request')
        clocks = []
        try:
            for _ in range(3):
                before = time.monotonic_ns()
                nvtx.nvtxMarkA(('handoff-clock:' + str(before)).encode())
                clocks.append([before, time.monotonic_ns()])
            result = run.request(prompt, 'profiled-repeat', 256)
            before = time.monotonic_ns()
            nvtx.nvtxMarkA(('handoff-clock:' + str(before)).encode())
            clocks.append([before, time.monotonic_ns()])
        finally:
            nvtx.nvtxRangePop()
        elapsed = time.monotonic() - t
        foreign = max(0, bench.busy_jiffies() - b - bench.proc_jiffies(pids) + o) / bench.HZ / elapsed
        (out / 'clocks.json').write_text(json.dumps(clocks) + '\n')
        run.event(kind='profiled_request', benchmark_valid=False, load_start=load, foreign_cores=foreign,
                  quiet=load < 2 and foreign < 1, **result)
        if a.engine == 'strata':
            run.event(kind='metrics', metrics=run.api('/metrics').get('history'))
    finally:
        try:
            stop_bpf(tracer)
        finally:
            run.stop()
            run.event(kind='released', gpu=run.gpu_apps())


def outer(a, out):
    out.mkdir(exist_ok=False)
    run = Run(out, time.time() + a.minutes * 60)
    proc = None
    try:
        run.quiet()
        run.held = True
        run.renew(force=True)
        bench.gw('POST', '/api/v1/gpu/free', {})
        assert not run.gpu_apps(), 'GPU has another owner'
        for name in ('tmp', 'config', 'cache'):
            (out / name).mkdir()
        env = dict(os.environ, TMPDIR=str(out / 'tmp'), XDG_CONFIG_HOME=str(out / 'config'), XDG_CACHE_HOME=str(out / 'cache'))
        cmd = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
               '--cuda-graph-trace=node', '--capture-range=nvtx', '--nvtx-capture=handoff-request',
               '--capture-range-end=stop', '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0',
               '--kill=none', '--duration=' + str(a.minutes * 60), '--export=none', '-o', str(out / 'trace'),
               sys.executable, str(Path(__file__).resolve()), '--engine', a.engine, '--out', a.out,
               '--minutes', str(a.minutes), '--driver']
        (out / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        with (out / 'nsys.txt').open('w') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                run.renew()
                if time.time() > run.end + 60:
                    raise TimeoutError('capture deadline')
                time.sleep(2)
        if proc.returncode:
            raise RuntimeError('capture driver failed; inspect logs')
    finally:
        if proc and proc.poll() is None:
            owner = out / 'owner.json'
            if owner.exists():
                pid = json.loads(owner.read_text())['pid']
                try:
                    argv = Path(f'/proc/{pid}/cmdline').read_bytes()
                    if b'capture.py' in argv and a.out.encode() in argv:
                        os.kill(pid, signal.SIGTERM)
                except FileNotFoundError:
                    pass
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=15)
        run.stop()
        run.event(kind='outer_released', gpu=run.gpu_apps())
    with (out / 'export.txt').open('w') as log:
        if not (out / 'trace.nsys-rep').exists():
            subprocess.run(['/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter', '-i', str(out / 'trace.qdstrm'),
                            '-o', str(out / 'trace.nsys-rep'), '-f'], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        subprocess.run(['nsys', 'export', '--type', 'sqlite', '-o', str(out / 'trace.sqlite'), str(out / 'trace.nsys-rep')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['strata', 'strata-reference'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--minutes', type=int, default=7)
    ap.add_argument('--driver', action='store_true')
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    (driver if args.driver else outer)(args, HERE / args.out)
