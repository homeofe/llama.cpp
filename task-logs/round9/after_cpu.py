#!/usr/bin/env python3
"""Screen existing DSpark confidence truncation after all five required stages."""
import json
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    while not (OUT / 'cpu-screen-complete.txt').exists():
        if time.time() >= 1792276260:  # 22:31 UTC; leave five minutes for this screen.
            print('confidence screen skipped: remaining budget reserved for completion', flush=True)
            return
        time.sleep(5)
    source = Path('/opt/llm-gateway/data/moe-quant/qwen38/round9-cpu-screen/runs.jsonl')
    assert json.loads(source.read_text().splitlines()[-1])['kind'] == 'all_complete'
    dest = Path('/opt/llm-gateway/data/moe-quant/qwen38/round9-dspark-confidence')
    with (OUT / 'bench-dspark-confidence.log').open('w') as stream:
        subprocess.run(['python3', '-u', 'task-logs/round9/bench.py', str(dest), '22:38',
                        '--rounds', '1', '--only', 'ling-dspark-conf50'], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert json.loads((dest / 'runs.jsonl').read_text().splitlines()[-1])['kind'] == 'all_complete'
    with (OUT / 'dspark-confidence-summary.json').open('w') as stream:
        subprocess.run(['python3', '-S', 'task-logs/round9/summarize.py', str(dest / 'runs.jsonl')],
                       cwd=ROOT, stdout=stream, check=True)
    (OUT / 'dspark-confidence-complete.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')
    print('confidence screen complete', flush=True)


if __name__ == '__main__':
    main()
