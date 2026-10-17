#!/usr/bin/env python3
"""Low-rate CPU-frequency and per-logical-CPU busy-time diagnosis."""
import json
from pathlib import Path
import sys
import time


def snapshot():
    rows = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            values = list(map(int, fields[1:9]))
            rows[int(fields[0][3:])] = (sum(values), sum(values) - values[3] - values[4])
    return rows


def main():
    pid = int(sys.argv[1])
    before = snapshot()
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        time.sleep(1)
        try:
            status = Path(f'/proc/{pid}/status').read_text()
        except FileNotFoundError:
            break
        after = snapshot()
        busy = {cpu: (b - before[cpu][1]) / max(1, t - before[cpu][0]) for cpu, (t, b) in after.items()}
        mhz = [int(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq').read_text()) / 1000
               for cpu in range(24)]
        memory = [line for line in status.splitlines() if line.startswith(('VmRSS:', 'VmSwap:', 'THP_enabled:'))]
        print(json.dumps(dict(epoch=time.time(), busy_by_cpu=busy, mhz_physical_cpus=mhz, memory=memory)), flush=True)
        before = after


if __name__ == '__main__':
    main()
