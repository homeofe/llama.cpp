#!/usr/bin/env python3
"""Temporarily disable proactive compaction only after explicit owner approval."""
import json
from pathlib import Path
import signal
import subprocess
import time

OUT = Path(__file__).resolve().parent
SETTING = Path('/proc/sys/vm/compaction_proactiveness')
DEADLINE = 1792277760  # 2026-10-17 22:56 UTC, the handoff deadline.


def set_value(value):
    subprocess.run(['sudo', '-n', 'sysctl', '-w', f'vm.compaction_proactiveness={value}'], check=True)
    if int(SETTING.read_text()) != value:
        raise RuntimeError('compaction setting verification failed')


def main():
    if time.time() >= DEADLINE:
        raise RuntimeError('measurement deadline has passed')
    previous = int(SETTING.read_text())
    record = dict(previous=previous, test_value=0, deadline_epoch=DEADLINE,
                  start_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    try:
        set_value(0)
        (OUT / 'compaction-active.json').write_text(json.dumps(record, indent=2) + '\n')
        while time.time() < DEADLINE and not (OUT / 'compaction-release.txt').exists():
            time.sleep(2)
    finally:
        set_value(previous)
        record['restored'] = int(SETTING.read_text())
        record['end_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        (OUT / 'compaction-restored.json').write_text(json.dumps(record, indent=2) + '\n')


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    main()
