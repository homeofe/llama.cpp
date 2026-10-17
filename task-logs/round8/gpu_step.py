#!/usr/bin/env python3
"""Run one local GPU step with a gateway lease and the sandbox load gate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

GW = 'http://127.0.0.1:8790/api/v1/gpu/'


def gateway(method, path, data=None):
    key = Path('/opt/llm-gateway/data/api-key').read_text().strip()
    request = urllib.request.Request(GW + path, method=method,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def apps():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('GPU unavailable: ' + result.stdout.strip() + result.stderr.strip())
    return {int(row.split(',')[0]): int(row.split(',')[1]) for row in result.stdout.splitlines() if row.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command required after --')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'accepted.txt').unlink(missing_ok=True)
    log = args.out / 'gate.jsonl'

    def event(**data):
        data['utc'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        with log.open('a') as stream:
            stream.write(json.dumps(data) + '\n')
        print(json.dumps(data), flush=True)

    try:
        apps()
    except (OSError, RuntimeError, ValueError) as error:
        event(kind='unavailable', reason=str(error))
        return 77
    for attempt in range(1, args.attempts + 1):
        deadline = time.monotonic() + 900
        while os.getloadavg()[0] >= 3:
            if time.monotonic() >= deadline:
                event(kind='timeout', reason='no quiet load window')
                return 75
            time.sleep(5)
        held = False
        child = None
        try:
            gateway('POST', 'free', {})
            gateway('POST', 'hold', {'reason': 'codex: round 8', 'ttlS': 900})
            held = True
            before_apps = apps()
            if any(mib > 300 for mib in before_apps.values()):
                raise RuntimeError('GPU has another process using more than 300 MiB')
            start_load = os.getloadavg()[0]
            if start_load >= 3:
                continue
            with (args.out / f'attempt-{attempt}.log').open('w') as output:
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                last_hold = time.monotonic()
                own_gpu_pid = None
                while child.poll() is None:
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        current = apps()
                        new_pids = set(current) - set(before_apps)
                        if own_gpu_pid is None and len(new_pids) == 1:
                            own_gpu_pid = next(iter(new_pids))
                        if any(mib > 300 for pid, mib in current.items() if pid != own_gpu_pid):
                            raise RuntimeError('another process now uses more than 300 MiB of VRAM')
                        if time.monotonic() - last_hold > 300:
                            gateway('POST', 'hold', {'reason': 'codex: round 8', 'ttlS': 900})
                            last_hold = time.monotonic()
                result = child.returncode
            end_load = os.getloadavg()[0]
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if held:
                gateway('DELETE', 'hold')
        accepted = result == 0 and start_load < 3 and end_load < 3
        event(kind='step', attempt=attempt, exit=result, load_start=start_load, load_end=end_load, accepted=accepted)
        if accepted:
            (args.out / 'accepted.txt').write_text(f'attempt-{attempt}.log\n')
            return 0
        if result not in (0, 75):
            return result
    return 75


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except (KeyboardInterrupt, OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
