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
    parser.add_argument('--plain', action='store_true')
    args = parser.parse_args()
    bench.configure()
    tag = args.prompt + ('-on' if args.graphs else '-off') + ('-plain' if args.plain else '')
    out = bench.ROOT / 'task-logs/round10' / ('profile-' + tag)
    out.mkdir(exist_ok=True)
    for name in ['tmp', 'config', 'cache']:
        (out / name).mkdir(exist_ok=True)
    target = ['python3', str(bench.ROOT / 'task-logs/round10/profile_request.py'), str(out), args.prompt]
    if args.graphs:
        target += ['--graphs']
    if args.plain:
        target += ['--plain']
        command = target
    else:
        command = ['sudo', '-n', 'env', 'TMPDIR=' + str(out / 'tmp'), 'XDG_CONFIG_HOME=' + str(out / 'config'),
                   'XDG_CACHE_HOME=' + str(out / 'cache'), 'nsys', 'profile', '--trace=cuda,nvtx,osrt',
                   '--sample=process-tree', '--cpuctxsw=process-tree', '--backtrace=dwarf', '--sampling-period=9140000',
                   '--capture-range=nvtx', '--nvtx-capture=round10', '--capture-range-end=stop',
                   '--kill=none', '--force-overwrite=true', '--export=sqlite', '-o', str(out / 'trace'),
                   'setpriv', '--reuid=' + str(os.getuid()), '--regid=' + str(os.getgid()), '--init-groups'] + target
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
        run.event(kind='profile_start', pid=proc.pid, graphs=args.graphs, plain=args.plain)
        while proc.poll() is None:
            run.renew()
            owned_file = out / 'owned-pids.json'
            owned = json.loads(owned_file.read_text()) if owned_file.exists() else {}
            foreign = {p: m for p, m in run.gpu_apps().items() if p not in owned.values() and m > 300}
            if foreign:
                raise RuntimeError('foreign GPU allocation ' + str(foreign))
            if time.time() > run.end:
                raise TimeoutError('profile deadline')
            time.sleep(3)
        run.event(kind='profile_exit', exit=proc.returncode)
        if proc.returncode:
            raise RuntimeError('profiler failed; inspect raw capture before retrying')
    finally:
        try:
            if proc is not None and proc.poll() is None:
                owned_file = out / 'owned-pids.json'
                if owned_file.exists():
                    owned = json.loads(owned_file.read_text())
                    for key in ['wrapper', 'server']:
                        try:
                            os.kill(owned[key], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                if args.plain:
                    proc.terminate()
                else:
                    subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', '-' + str(proc.pid)], check=False)
                proc.wait(timeout=30)
        finally:
            run.stop()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
