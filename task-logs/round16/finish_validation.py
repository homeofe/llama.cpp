#!/usr/bin/env python3
"""Run analysis and the two quality panels only after the confirmation releases its GPU."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
child = None

def event(phase, **fields):
    print(json.dumps(dict(phase=phase, utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), **fields)), flush=True)

def stop(*_):
    raise KeyboardInterrupt('SIGTERM')

signal.signal(signal.SIGTERM, stop)

def run(argv):
    global child
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    code = child.wait()
    child = None
    if code:
        raise RuntimeError(f'validation subprocess returned {code}: {argv[1]}')

try:
    event('waiting_for_confirmation')
    deadline = time.time()+85*60
    while time.time() < deadline:
        log = here/'confirm-clean/runs.jsonl'
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        if rows and rows[-1]['kind'] == 'released':
            if not any(row['kind'] == 'complete' for row in rows):
                raise RuntimeError('confirmation ended without completion')
            break
        time.sleep(3)
    else:
        raise TimeoutError('confirmation deadline')
    event('analyze')
    run([sys.executable, str(here/'analyze.py'), str(here/'confirm-clean')])
    report = json.loads((here/'confirm-clean/summary.json').read_text())
    passed = all(report['categories'][category][metric+'_gain_pct'] >= 3
                 for category in ('code','en') for metric in ('stream','server'))
    if not passed:
        event('performance_gate_failed')
        sys.exit(0)
    for name in ('base','q6gpu'):
        event('quality_start', config=name)
        run([sys.executable, '-u', str(here/'quality.py'), name, 'quality-'+name])
        event('quality_complete', config=name,
              verdict=json.loads((here/('quality-'+name)/'panel-verdict.json').read_text()))
    event('validation_complete')
finally:
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=90)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
