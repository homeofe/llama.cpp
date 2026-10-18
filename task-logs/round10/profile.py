#!/usr/bin/env python3
"""Nsight CPU/CUDA timeline of complete MTP requests, with a separately owned GPU lease."""
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
    parser.add_argument('prompt', choices=['code-duration', 'en-1'])
    parser.add_argument('--graphs', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plain', action='store_true')
    mode.add_argument('--counters-only', action='store_true')
    args = parser.parse_args()
    bench.configure()
    tag = args.prompt + ('-on' if args.graphs else '-off') + ('-plain' if args.plain else '-counters' if args.counters_only else '')
    out = bench.ROOT / 'task-logs/round10' / ('profile-' + tag)
    out.mkdir(exist_ok=True)
    if (out / 'owned-pids.json').exists():
        raise RuntimeError('capture directory already used; archive it before a retry')
    for name in ['tmp', 'config', 'cache']:
        (out / name).mkdir(exist_ok=True)
    target = ['python3', str(bench.ROOT / 'task-logs/round10/profile_request.py'), str(out), args.prompt]
    if args.graphs:
        target += ['--graphs']
    direct = args.plain or args.counters_only
    if direct:
        target += ['--plain' if args.plain else '--counters-only']
        command = target
    else:
        command = ['sudo', '-n', 'env', 'TMPDIR=' + str(out / 'tmp'), 'XDG_CONFIG_HOME=' + str(out / 'config'),
                   'XDG_CACHE_HOME=' + str(out / 'cache'), 'nsys', 'profile', '--trace=cuda,nvtx,osrt',
                   '--sample=process-tree', '--cpuctxsw=process-tree', '--backtrace=dwarf',
                   '--sampling-period=18000000', '--samples-per-backtrace=1',
                   '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0',
                   '--capture-range=nvtx', '--nvtx-capture=round10', '--capture-range-end=stop',
                   '--kill=none', '--force-overwrite=true', '--export=sqlite', '-o', str(out / 'trace')] + target
    (out / 'command.json').write_text(json.dumps(dict(command=command, permission='Owner explicitly authorized sudo and CPU sampling',
        production_source_changed=False, no_build=True), indent=2) + '\n')
    run = bench.Run(out, min(time.time() + 1500, 1792320900))
    proc = None
    try:
        run.quiet()
        bench.stage.gw('POST', '/api/v1/gpu/free', {})
        run.held = True
        run.renew(force=True)
        if any(m > 300 for m in run.gpu_apps().values()):
            raise RuntimeError('foreign GPU allocation')
        with (out / 'profiler.log').open('w') as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'profiler-owner.json').write_text(json.dumps(dict(pid=proc.pid)) + '\n')
        run.event(kind='profile_start', pid=proc.pid, graphs=args.graphs, plain=args.plain, counters_only=args.counters_only)
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
        if proc.returncode:
            raise RuntimeError('profiler failed; inspect raw capture before retrying')
        run.stop()
        if not direct and not (out / 'trace.sqlite').exists():
            if not (out / 'trace.qdstrm').exists():
                raise RuntimeError('capture range never produced a trace')
            with (out / 'import.log').open('w') as stream:
                subprocess.run(['/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter',
                    '-i', str(out / 'trace.qdstrm'), '-o', str(out / 'trace.nsys-rep'), '-f'],
                    stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
                subprocess.run(['nsys', 'export', '--type', 'sqlite', '--force-overwrite=true',
                    '-o', str(out / 'trace.sqlite'), str(out / 'trace.nsys-rep')],
                    stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
    finally:
        try:
            owned_file = out / 'owned-pids.json'
            if owned_file.exists():
                owned = json.loads(owned_file.read_text())
                for key in ['wrapper', 'server']:
                    try:
                        actual = Path(f'/proc/{owned[key]}/cmdline').read_bytes().split(b'\0')
                    except FileNotFoundError:
                        continue
                    expected = str(bench.ROOT / ('task-logs/round10/profile_request.py' if key == 'wrapper' else 'build-cuda/bin/llama-server')).encode()
                    if expected in actual:
                        if direct:
                            try:
                                os.kill(owned[key], signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        else:
                            subprocess.run(['sudo', '-n', 'kill', '-TERM', str(owned[key])], check=False)
            if proc is not None and proc.poll() is None:
                if direct:
                    proc.terminate()
                else:
                    subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', '-' + str(proc.pid)], check=False)
                proc.wait(timeout=30)
        finally:
            run.stop()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
