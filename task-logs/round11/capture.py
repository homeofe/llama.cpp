#!/usr/bin/env python3
"""Capture CUDA copy/compute overlap; the outer process owns the GPU lease."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=['qwen-control', 'qwen-async', 'ling-control', 'ling-async'])
    args = parser.parse_args()
    bench.configure()
    out = bench.ROOT / 'task-logs/round11' / ('capture-' + args.config)
    out.mkdir(exist_ok=True)
    if (out / 'profiler-owner.json').exists():
        raise RuntimeError('capture directory already used')
    for name in ['tmp', 'config', 'cache', 'cuda-cache']:
        (out / name).mkdir(exist_ok=True)
    command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
        '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0', '--capture-range=nvtx', '--nvtx-capture=round11',
        '--capture-range-end=stop', '--kill=none', '--force-overwrite=true', '--export=sqlite',
        '-o', str(out / 'trace'), 'python3', str(bench.ROOT / 'task-logs/round11/capture_request.py'), str(out), args.config]
    env = {**os.environ, 'TMPDIR': str(out / 'tmp'), 'XDG_CONFIG_HOME': str(out / 'config'),
        'XDG_CACHE_HOME': str(out / 'cache'), 'CUDA_CACHE_PATH': str(out / 'cuda-cache')}
    (out / 'command.json').write_text(json.dumps(dict(command=command, cpu_sampling=False,
        graphs=False, purpose='Diagnostic only: instrumented timing is not the adoption benchmark.'), indent=2) + '\n')
    run = bench.Run(out, min(time.time() + 900, 1792335540))
    proc = None
    try:
        run.quiet()
        bench.stage.gw('POST', '/api/v1/gpu/free', {})
        run.held = True
        run.renew(force=True)
        if any(m > 300 for m in run.gpu_apps().values()):
            raise RuntimeError('foreign GPU allocation')
        with (out / 'profiler.log').open('w') as stream:
            proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'profiler-owner.json').write_text(json.dumps(dict(pid=proc.pid)) + '\n')
        while proc.poll() is None:
            if run.held and (out / 'server-stopped.json').exists():
                run.stop()
                run.event(kind='gpu_released_before_trace_export')
            run.renew()
            owned_file = out / 'owned-pids.json'
            owned = json.loads(owned_file.read_text()) if owned_file.exists() else {}
            foreign = {p: m for p, m in run.gpu_apps().items() if p not in owned.values() and m > 300}
            if run.held and foreign:
                raise RuntimeError('foreign GPU allocation ' + str(foreign))
            if time.time() > run.end:
                raise TimeoutError('profile deadline')
            time.sleep(3)
        run.event(kind='profile_exit', exit=proc.returncode)
        run.stop()
        if proc.returncode:
            raise RuntimeError('profiler failed; raw capture retained')
        if not (out / 'trace.sqlite').exists():
            with (out / 'import.log').open('w') as stream:
                subprocess.run(['/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter', '-i', str(out / 'trace.qdstrm'),
                    '-o', str(out / 'trace.nsys-rep'), '-f'], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
                subprocess.run(['nsys', 'export', '--type', 'sqlite', '--force-overwrite=true', '-o', str(out / 'trace.sqlite'),
                    str(out / 'trace.nsys-rep')], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
    finally:
        try:
            owned_file = out / 'owned-pids.json'
            if owned_file.exists():
                for role, pid in json.loads(owned_file.read_text()).items():
                    expected = (str(bench.ROOT / 'task-logs/round11/capture_request.py') if role == 'wrapper'
                        else bench.stage.CONFIGS[args.config]['cmd'][0]).encode()
                    try:
                        if expected in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
                            os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except FileNotFoundError:
                        pass
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
        finally:
            run.stop()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
