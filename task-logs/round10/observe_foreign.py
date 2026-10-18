#!/usr/bin/env python3
"""Attribute transient foreign CPU load without changing system settings."""
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('observe_cpu', ROOT / 'task-logs/round9/observe_cpu.py')
assert spec and spec.loader
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
previous = observer.snapshot()
with (ROOT / 'task-logs/round10/foreign-processes.jsonl').open('a') as stream:
    for _ in range(300):
        time.sleep(1)
        current = observer.snapshot()
        elapsed = current[0] - previous[0]
        deltas = []
        for pid, (name, ticks) in current[2].items():
            if pid in previous[2]:
                delta = ticks - previous[2][pid][1]
                if delta > 0:
                    deltas.append(dict(pid=pid, name=name, cores=delta / observer.HZ / elapsed))
        deltas.sort(key=lambda x: x['cores'], reverse=True)
        stream.write(json.dumps(dict(epoch=time.time(), elapsed_s=elapsed, top=deltas[:12])) + '\n')
        stream.flush()
        previous = current
