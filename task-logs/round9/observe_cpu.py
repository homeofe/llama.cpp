#!/usr/bin/env python3
"""Record signed CPU accounting deltas and the processes responsible for load."""
import collections
import json
import os
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
LOG = OUT / 'ling-off-ppl/attempt-3.log'
HZ = os.sysconf('SC_CLK_TCK')


def snapshot():
    busy = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    processes = {}
    for directory in Path('/proc').iterdir():
        if directory.name.isdigit():
            try:
                raw = (directory / 'stat').read_text()
                name, rest = raw.split('(', 1)[1].rsplit(')', 1)
                fields = rest.split()
                processes[int(directory.name)] = (name, int(fields[11]) + int(fields[12]))
            except (OSError, ValueError):
                pass
    return time.monotonic(), busy, processes


def main():
    deadline = time.monotonic() + 600
    while 'calculating perplexity over' not in LOG.read_text():
        if time.monotonic() > deadline:
            raise TimeoutError('evaluation did not start')
        time.sleep(.25)
    previous = initial = snapshot()
    pids = [pid for pid, (name, _) in initial[2].items() if name.startswith('llama-perplexit')]
    assert len(pids) == 1, pids
    own_pid = pids[0]
    totals = collections.Counter()
    names = {}
    signed = clamped = 0
    samples = []
    while 'Final estimate: PPL =' not in LOG.read_text():
        time.sleep(1)
        current = snapshot()
        t, busy, procs = current
        pt, old_busy, old_procs = previous
        busy_delta = sum(busy[:8]) - busy[3] - busy[4] - (sum(old_busy[:8]) - old_busy[3] - old_busy[4])
        own = procs[own_pid][1] - old_procs[own_pid][1]
        signed += busy_delta - own
        clamped += max(0, busy_delta - own)
        for pid, (name, ticks) in procs.items():
            if pid in old_procs:
                totals[pid] += max(0, ticks - old_procs[pid][1])
                names[pid] = name
        samples.append(dict(wall_s=t - pt, busy_ticks=busy_delta, own_ticks=own, foreign_signed_ticks=busy_delta - own))
        previous = current
    wall = previous[0] - initial[0]
    result = dict(own_pid=own_pid, wall_s=wall, signed_foreign_cores=signed/HZ/wall,
                  per_interval_clamped_foreign_cores=clamped/HZ/wall,
                  top_processes=[dict(pid=pid, name=names[pid], cores=ticks/HZ/wall) for pid, ticks in totals.most_common(20)],
                  samples=samples)
    (OUT / 'cpu-accounting.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
