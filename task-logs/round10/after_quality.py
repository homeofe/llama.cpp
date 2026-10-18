#!/usr/bin/env python3
"""Start the selected panel only after quality released the GPU."""
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'


def main():
    while not (OUT / 'quality-queue-complete.json').exists():
        if time.time() > 1792315560:
            raise TimeoutError('quality did not complete; panel was not started')
        time.sleep(3)
    with (OUT / 'quality-summary.json').open('w') as stream:
        subprocess.run(['python3', str(OUT / 'quality_summary.py')], cwd=ROOT, stdout=stream, check=True)
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + ' Round 10: Ling-Qualitaetschecks abgeschlossen; drei Hard-/Agent-Panelrunden mit DSpark n3/p0 starten.\n')
    process = None
    try:
        with (OUT / 'panels-driver.log').open('w') as stream:
            process = subprocess.Popen(['python3', '-u', str(OUT / 'panels.py'), 'ling-dspark-n3-p00'], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
        code = process.wait()
        (OUT / 'panels-complete.json').write_text(json.dumps(dict(exit=code, epoch=time.time())) + '\n')
        if code:
            raise RuntimeError('panel failed; inspect panels-driver.log')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=70)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
