#!/usr/bin/env python3
"""Run the bounded CPU screen only after the explicit multiplicity decision."""
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def command(args, log, required=True):
    with (OUT / log).open('w') as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(log, result.returncode, flush=True)
    if required and result.returncode:
        raise RuntimeError(log)
    return result.returncode


def main():
    assert (OUT / 'multiplicity.json').exists(), 'complete task 4 first'
    assert (OUT / 'cpu-kernel-decision.json').exists(), 'record the bound decision first'
    decision = json.loads((OUT / 'cpu-kernel-decision.json').read_text())
    assert decision['proceed_to_cpu_sweep'], 'unresolved kernel work'
    command(['python3', '-S', 'task-logs/round9/sweep_configs.py'], 'cpu-configs.log')
    names = json.loads((OUT / 'cpu-sweep-manifest.json').read_text())['variants']
    native = command(['python3', '-S', 'task-logs/round9/native_pool.py'], 'native-pool.log', False)
    if native == 0:
        names.append('qwen-cpu-native-pool')
    events = ','.join(f'amd_df/dram_channel_data_controller_{i}/' for i in [2, 3, 4, 5])
    counters = ['sudo', '-n', 'perf', 'stat', '-a', '-I', '1000', '-x', ',', '-e', events,
                '--', 'sleep', '1500']
    with (OUT / 'dram-cpu-screen.csv').open('w') as stream:
        epoch = time.time()
        proc = subprocess.Popen(counters, stdout=subprocess.DEVNULL, stderr=stream, start_new_session=True)
    (OUT / 'dram-cpu-screen-meta.json').write_text(json.dumps(dict(start_epoch=epoch,
        command=counters, owned_process_group=proc.pid, channels=[2, 3, 4, 5], bytes_per_event=64), indent=2) + '\n')
    try:
        command(['python3', '-u', 'task-logs/round9/bench.py',
                 '/opt/llm-gateway/data/moe-quant/qwen38/round9-cpu-screen', '22:45',
                 '--rounds', '1', '--only'] + names, 'bench-cpu-screen.log')
    finally:
        if proc.poll() is None:
            subprocess.run(['sudo', '-n', 'kill', '-INT', '--', '-' + str(proc.pid)], check=False)
            proc.wait(timeout=15)
    command(['python3', '-S', 'task-logs/round9/summarize.py',
             '/opt/llm-gateway/data/moe-quant/qwen38/round9-cpu-screen/runs.jsonl'], 'cpu-screen-summary.log')
    (OUT / 'cpu-screen-complete.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    main()
