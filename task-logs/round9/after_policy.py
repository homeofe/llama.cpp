#!/usr/bin/env python3
"""Finish task 1 after its interleaved server run has released the GPU."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'
RUNS = Path('/opt/llm-gateway/data/moe-quant/qwen38/round9-policy/runs.jsonl')


def command(args, name, expected=0):
    with (OUT / name).open('w') as output:
        result = subprocess.run(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode != expected:
        raise RuntimeError(f'{name}: exit {result.returncode}, expected {expected}')


def main():
    deadline = time.monotonic() + 5400
    while True:
        rows = [json.loads(line) for line in RUNS.read_text().splitlines()]
        if rows[-1]['kind'] == 'all_complete':
            requests = [r for r in rows if r.get('kind') == 'request' and r.get('accepted')]
            if len({r['label'] for r in requests}) != 96:
                raise RuntimeError('benchmark incomplete')
            break
        if rows[-1]['kind'] == 'stop' or time.monotonic() > deadline:
            raise RuntimeError('benchmark stopped or timed out')
        time.sleep(10)
    command(['nice', '-n', '10', 'cmake', '--build', 'build-cpu', '--target', 'test-moe-cache', '-j', '8'], 'build-histogram.log')
    command(['nice', '-n', '10', 'c++', '-O2', '-std=c++17', '-Iinclude', '-Isrc', '-Icommon', '-Iggml/include',
             '-Ivendor', '-Ivendor/nlohmann', 'task-logs/round5/moe-probe.cpp', '-Lbuild-cuda/bin',
             '-Wl,-rpath,' + str(ROOT / 'build-cuda/bin'), '-lllama-common', '-lllama', '-lggml', '-lggml-base',
             '-o', 'task-logs/round9/moe-probe'], 'build-probe.log')
    command(['python3', '-S', 'task-logs/round9/check_replay.py', str(OUT / 'replay-before-hist')], 'metric-old.log', expected=1)
    command(['python3', '-S', 'task-logs/round9/check_replay.py'], 'metric-new.log')
    for model in ['qwen', 'ling']:
        for policy in ['off', 'off-knobs']:
            command(['python3', '-S', 'task-logs/round9/quality.py', model, policy, '--probe'], f'{model}-{policy}-probe-run.log')
        a = OUT / f'{model}-off-probe/logits.logits.bin'
        b = OUT / f'{model}-off-knobs-probe/logits.logits.bin'
        command(['cmp', str(a), str(b)], f'{model}-off-bitwise.log')
        for policy in ['off', 'r7', 'policy']:
            command(['python3', '-S', 'task-logs/round9/quality.py', model, policy], f'{model}-{policy}-ppl-run.log')
        with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
            stream.write(time.strftime('%H:%M:%S') + f' Runde 9: {model} Cache-off bitgenau; alle drei Perplexity-Messungen abgeschlossen, Auswertung folgt.\n')
    (OUT / 'quality-complete.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')


if __name__ == '__main__':
    main()
