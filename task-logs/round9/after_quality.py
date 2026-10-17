#!/usr/bin/env python3
"""Start profiling and allocation probes after task 1 releases its lease."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def command(args, log):
    with (OUT / log).open('w') as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(log, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(log)


def main():
    deadline = time.monotonic() + 2700
    while not (OUT / 'quality-complete.txt').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('quality suite has not completed')
        time.sleep(5)
    command(['python3', '-S', 'task-logs/round9/quality_summary.py'], 'quality-summary.log')
    report = json.loads((OUT / 'quality-summary.json').read_text())
    assert len(report['perplexity']) == 6
    assert all(row['within_half_percent'] for row in report['perplexity'])
    assert len(report['probes']) == 2 and all(row['bit_exact'] for row in report['probes'])
    command(['nice', '-n', '10', 'c++', '-O2', '-std=c++17', '-fPIC', '-shared', '-Iggml/include',
             'task-logs/round9/cpu-profile.cpp', '-ldl', '-o', 'task-logs/round9/cpu-profile.so'], 'build-cpu-profile.log')
    command(['build-cpu/bin/test-moe-cache'], 'cache-unit.log')
    command(['python3', '-S', 'task-logs/round9/profile.py', 'q-warm-draft', '--perf'], 'profile-run.log')
    command(['python3', '-S', 'task-logs/round9/fit.py', 'qwen', '2944'], 'fit-qwen.log')
    command(['python3', '-S', 'task-logs/round9/fit.py', 'ling', '6144'], 'fit-ling.log')
    (OUT / 'fit-complete.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')


if __name__ == '__main__':
    main()
