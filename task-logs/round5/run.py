#!/usr/bin/env python3
"""Run one GPU experiment with a gateway hold and foreign-load audit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'task-logs/round5'
DEADLINE = 1792253250  # 2026-10-17 16:07:30 UTC


def gateway(method, endpoint, payload=None):
    key = Path('/opt/llm-gateway/data/api-key').read_text().strip()
    request = urllib.request.Request('http://127.0.0.1:8790/api/v1/gpu/' + endpoint,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def snapshot():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            end = text.rindex(')')
            fields = text[end + 2:].split()
            result[int(path.parent.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]), text[text.index('(') + 1:end])
        except (OSError, ValueError, IndexError):
            pass
    return result


def foreign(before, after, seconds, own):
    own = set(own)
    for _ in range(12):
        own.update(pid for pid, (parent, _, _) in after.items() if parent in own)
    hz = os.sysconf('SC_CLK_TCK')
    return [{'pid': pid, 'name': row[2], 'cpu': (row[1] - before[pid][1]) / hz / seconds}
            for pid, row in after.items() if pid in before and pid not in own
            and (row[1] - before[pid][1]) / hz / seconds > .5]


def interrupted(signum, frame):
    raise InterruptedError(f'signal {signum}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path('/proc/1/comm').read_text().strip() not in ('systemd', 'init'):
        raise RuntimeError('host process visibility is required for the foreign-load audit')
    signal.signal(signal.SIGTERM, interrupted)
    for attempt in range(1, 5):
        prefix = LOG / f'{args.label}-a{attempt}'
        if prefix.with_suffix('.meta.json').exists():
            raise RuntimeError(f'existing evidence: {prefix}')
        while True:
            if time.time() >= DEADLINE:
                raise RuntimeError('round budget expired')
            before = snapshot()
            start = time.monotonic()
            time.sleep(3)
            busy = foreign(before, snapshot(), time.monotonic() - start, {os.getpid()})
            if os.getloadavg()[0] < 3 and not busy:
                break
            print('WAIT', args.label, os.getloadavg(), busy, flush=True)
            time.sleep(10)
        held = False
        child = None
        meta = {'command': args.command, 'start': time.time(), 'load': os.getloadavg(),
                'environment': {k: v for k, v in os.environ.items() if k.startswith(('GGML_', 'LLAMA_', 'OMP_'))}, 'foreign': []}
        try:
            gateway('POST', 'free', {})
            gateway('POST', 'hold', {'reason': 'codex: round 5 cache', 'ttlS': 900})
            held = True
            last_hold = time.monotonic()
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
            meta['gpu_before'] = gpu
            if gpu.strip():
                raise RuntimeError('GPU occupied: ' + gpu)
            print('START', prefix.name, flush=True)
            with prefix.with_suffix('.out').open('w') as out, prefix.with_suffix('.err').open('w') as err:
                before = snapshot()
                start = time.monotonic()
                child = subprocess.Popen(args.command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
                meta['pid'] = child.pid
                while child.poll() is None:
                    time.sleep(3)
                    after = snapshot()
                    now = time.monotonic()
                    meta['foreign'].extend(foreign(before, after, now - start, {os.getpid(), child.pid}))
                    before, start = after, now
                    if now - last_hold > 300:
                        gateway('POST', 'hold', {'reason': 'codex: round 5 cache', 'ttlS': 900})
                        last_hold = now
                    if time.time() >= DEADLINE:
                        raise RuntimeError('round budget expired')
            meta['returncode'] = child.returncode
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if held:
                gateway('DELETE', 'hold')
            meta['end'] = time.time()
            prefix.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2) + '\n')
        print('END', prefix.name, 'rc', meta['returncode'], 'foreign', meta['foreign'], flush=True)
        if meta['returncode']:
            raise SystemExit(meta['returncode'])
        if not meta['foreign']:
            return
    raise RuntimeError('foreign load in all attempts')


if __name__ == '__main__':
    main()
