#!/usr/bin/env python3
"""Start the pinned Ling draft experiment only after the slot comparison finishes."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def command(args, log):
    with (OUT / log).open('w') as stream:
        subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(log, 'complete', flush=True)


def main():
    deadline = min(time.time() + 3600, 1792273860)  # 21:51 UTC latest start.
    while not (OUT / 'slots-complete.txt').exists():
        if time.time() > deadline:
            raise TimeoutError('slot comparison has not completed')
        time.sleep(5)
    rows = [json.loads(line) for line in Path('/opt/llm-gateway/data/moe-quant/qwen38/round9-slots/runs.jsonl').read_text().splitlines()]
    assert rows[-1]['kind'] == 'all_complete', 'slot screening incomplete'
    command(['python3', '-S', 'task-logs/round9/prepare_dspark.py'], 'dspark-prepare.log')
    command(['python3', '-S', 'task-logs/round9/dspark_fit.py'], 'dspark-fit.log')
    command(['python3', '-u', 'task-logs/round9/bench.py',
             '/opt/llm-gateway/data/moe-quant/qwen38/round9-dspark', '22:15', '--rounds', '1',
             '--only', 'ling-cache-mtp3', 'ling-dspark-off', 'ling-dspark-cache'], 'bench-dspark.log')
    command(['python3', '-S', 'task-logs/round9/summarize.py',
             '/opt/llm-gateway/data/moe-quant/qwen38/round9-dspark/runs.jsonl'], 'dspark-summary.log')
    (OUT / 'dspark-complete.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')


if __name__ == '__main__':
    main()
