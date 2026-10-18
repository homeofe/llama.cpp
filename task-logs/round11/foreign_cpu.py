#!/usr/bin/env python3
"""Read-only per-process accounting to identify repeated foreign-gate failures."""
import json
import os
from pathlib import Path
import time


def snapshot():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            fields = text.rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = (int(fields[11]) + int(fields[12]),
                text[text.index('(') + 1:text.rindex(')')], int(fields[1]))
        except (OSError, IndexError, ValueError):
            pass
    return result


if __name__ == '__main__':
    out = Path(__file__).with_name('foreign-cpu.jsonl')
    previous = snapshot()
    last = time.monotonic()
    end = last + 1500
    print(json.dumps(dict(pid=os.getpid())), flush=True)
    while time.monotonic() < end:
        time.sleep(1)
        current = snapshot()
        now = time.monotonic()
        rows = []
        for pid, (ticks, name, parent) in current.items():
            if pid in previous:
                cores = (ticks - previous[pid][0]) / os.sysconf('SC_CLK_TCK') / (now - last)
                if cores >= 0.05:
                    rows.append(dict(pid=pid, name=name, parent=parent, cores=cores))
        with out.open('a') as stream:
            stream.write(json.dumps(dict(epoch=time.time(), seconds=now-last, processes=sorted(rows, key=lambda r: -r['cores']))) + '\n')
        previous, last = current, now
