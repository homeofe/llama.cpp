#!/usr/bin/env python3
"""Finish the real-trace bound after the drafter stage, then inspect graph launches."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def main():
    while not (OUT / 'dspark-complete.txt').exists():
        if time.time() > 1792275600:  # 22:20 UTC; preserve time for the CPU decision.
            raise TimeoutError('drafter stage has not completed')
        time.sleep(5)
    rows = [json.loads(line) for line in Path('/opt/llm-gateway/data/moe-quant/qwen38/round9-dspark/runs.jsonl').read_text().splitlines()]
    assert rows[-1]['kind'] == 'all_complete', 'drafter comparison incomplete'
    with (OUT / 'multiplicity.log').open('w') as stream:
        subprocess.run(['python3', '-S', 'task-logs/round9/multiplicity.py'], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('multiplicity complete', flush=True)
    with (OUT / 'nsys-graphs.log').open('w') as stream:
        result = subprocess.run(['python3', '-S', 'task-logs/round9/gpu_profile.py', 'nsys', '--graphs'],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print('graph trace exit', result.returncode, flush=True)
    (OUT / 'cpu-decision-ready.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')


if __name__ == '__main__':
    main()
