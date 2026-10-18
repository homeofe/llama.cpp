#!/usr/bin/env python3
"""Run bounded quality checks after the complete confirmation has released the GPU."""
import json
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'


def main():
    while not (OUT / 'confirmation-complete.json').exists():
        if time.time() > 1792314600:  # 09:10 UTC
            raise TimeoutError('confirmation not completed; quality was not started')
        time.sleep(5)
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + ' Round 10: Bestaetigung abgeschlossen; frische Bitgleichheits- und Perplexity-Pruefungen starten.\n')
    jobs = [
        ['ling', 'off-knobs', '--probe', '--attempts', '1'],
        ['ling', 'off', '--attempts', '1'],
        ['ling', 'r7', '--budget-mib', '4608', '--attempts', '1'],
    ]
    results = []
    process = None
    try:
        for index, arguments in enumerate(jobs):
            remaining = 1792315500 - time.time()  # 09:25 UTC
            if remaining <= 0:
                raise TimeoutError('quality queue deadline')
            command = ['python3', '-u', str(OUT / 'quality.py')] + arguments
            with (OUT / f'quality-{index}.log').open('w') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            code = process.wait(timeout=remaining)
            results.append(dict(command=command, exit=code))
            print(json.dumps(results[-1]), flush=True)
            if code not in [0, 75]:
                raise RuntimeError('quality command failed')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=65)
        (OUT / 'quality-queue-results.json').write_text(json.dumps(results, indent=2) + '\n')
    (OUT / 'quality-queue-complete.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))) + '\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
