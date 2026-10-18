#!/usr/bin/env python3
"""Run the six whole-request Qwen measurements after the isolated panels."""
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'


def main():
    while not (OUT / 'panels-complete.json').exists():
        if time.time() > 1792318260:
            raise TimeoutError('panels did not complete; profiles were not started')
        time.sleep(3)
    assert json.loads((OUT / 'panels-complete.json').read_text())['exit'] == 0
    with (OUT / 'panel-summary-driver.log').open('w') as stream:
        subprocess.run(['python3', str(OUT / 'panel_summary.py')], cwd=ROOT, stdout=stream, check=True)
    assert json.loads((OUT / 'panels/summary.json').read_text())['complete']
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + ' Round 10: Ling-Panels abgeschlossen; Qwen-Gesamtrundenprofil startet (etwa 15-25 Minuten, sechs getrennte Serverlaeufe).\n')
    jobs = [[prompt] + mode for mode in [[], ['--graphs', '--counters-only'], ['--graphs', '--plain']]
            for prompt in ['code-duration', 'en-1']]
    results = []
    process = None
    try:
        for index, arguments in enumerate(jobs):
            command = ['python3', '-u', str(OUT / 'profile.py')] + arguments
            with (OUT / f'profile-driver-{index}.log').open('w') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            code = process.wait()
            results.append(dict(command=command, exit=code, epoch=time.time()))
            (OUT / 'profile-queue-results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(results[-1]), flush=True)
            if code:
                raise RuntimeError('profile failed; inspect corresponding driver log')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=70)
    (OUT / 'profile-queue-complete.json').write_text(json.dumps(dict(epoch=time.time())) + '\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
