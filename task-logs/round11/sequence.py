#!/usr/bin/env python3
"""Continue independent measurement stages after the existing Qwen series exits."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'
child = None


def record(**event):
    with (OUT / 'sequence.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(epoch=time.time(), **event)) + '\n')


def invoke(name, command):
    global child
    record(kind='start', name=name, command=command)
    with (OUT / (name + '-driver.log')).open('w') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record(kind='process', name=name, pid=child.pid)
        result = child.wait()
    record(kind='exit', name=name, code=result)
    if result:
        raise RuntimeError(name + ' failed: ' + str(result))
    child = None


def main():
    record(kind='sequence_start', pid=os.getpid())
    while Path('/proc/101574').exists():
        if time.time() > 1792329300:
            raise TimeoutError('Qwen prerequisite exceeded 13:15 UTC')
        time.sleep(10)
    rows = [json.loads(line) for line in (OUT / 'qwen-e2e/runs.jsonl').read_text().splitlines()]
    assert rows[-1]['kind'] == 'all_complete', 'Qwen did not complete'
    invoke('qwen-summary', ['python3', str(OUT / 'e2e_summary.py'), 'qwen'])
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()) + ' Runde 11: Qwen drei Vergleichsrunden fertig; Ling beginnt, danach Bitgleichheit und Perplexity.\n')
    invoke('ling-e2e', ['python3', str(OUT / 'bench.py'), str(OUT / 'ling-e2e'), '13:30',
        '--only', 'ling-control', 'ling-async', '--rounds', '3'])
    invoke('ling-summary', ['python3', str(OUT / 'e2e_summary.py'), 'ling'])
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()) + ' Runde 11: Ling drei Vergleichsrunden fertig; Qualitätsprüfung beider Modelle beginnt.\n')
    for model in ['qwen', 'ling']:
        for policy, probe in [('off', True), ('off', False), ('r7', False)]:
            name = model + '-' + policy + '-async-' + ('probe' if probe else 'ppl')
            command = ['python3', str(OUT / 'quality.py'), model, policy, '--tag', 'async',
                '--async-upload', '--runtime-dir', str(OUT / 'candidate-bin')]
            if probe:
                command.append('--probe')
            if model == 'ling' and policy == 'r7':
                command += ['--budget-mib', '4608']
            invoke(name, command)
    record(kind='sequence_complete')


def interrupt(*_):
    if child is not None and child.poll() is None:
        child.terminate()
        child.wait(timeout=90)
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt)
    main()
