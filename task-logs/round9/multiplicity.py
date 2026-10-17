#!/usr/bin/env python3
import collections
import json
import os
from pathlib import Path
import re
import subprocess
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'
TRACES = Path('/opt/llm-gateway/data/moe-quant/traces')
PROFILE = '/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1-2048mib.profile'


class Result(TypedDict, total=False):
    trace: str
    policy: str
    phase: str
    quant: str
    histogram: dict[int, int]
    unique_matrices: int
    mean: float
    distinct_weight_bytes: int
    routed_weight_bytes: int
    byte_weighted_mean: float


def main():
    inventory = json.loads((ROOT / 'task-logs/round8/model-inventory.json').read_text())
    results: list[Result] = []
    for trace, model in [('qwen38-bench8', 'qwen38'), ('ling-bench8', 'ling'), ('qwen38-long-de', 'qwen38')]:
        source = TRACES / f'{trace}.jsonl'
        with source.open() as stream:
            target = next(b['context'] for line in stream if not (b := json.loads(line)).get('draft', True) and 'event' not in b)
        for policy in ['r7', 'byte-decay64']:
            warm = trace == 'qwen38-bench8'
            env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_MOE_')}
            env.update(GGML_MOE_REPLAY_MISS_HISTOGRAM='1', GGML_MOE_REPLAY_WARM=str(int(warm)),
                       GGML_MOE_HYBRID_BYTE_VALUE=str(int(policy != 'r7')))
            cmd = [str(ROOT / 'build-cpu/bin/test-moe-cache'), '--replay-log', str(source), '0', '32', '1024', '0',
                   '128' if policy == 'r7' else '64', PROFILE if warm else '-', str(target)]
            result = subprocess.check_output(cmd, env=env, text=True)
            (OUT / f'multiplicity-{trace}-{policy}.log').write_text(result)
            histograms: dict[tuple[str, str], collections.Counter[int]] = collections.defaultdict(collections.Counter)
            weights: dict[tuple[str, str], collections.Counter[int]] = collections.defaultdict(collections.Counter)
            triplets: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
            for line in result.splitlines():
                if not line.startswith('miss_histogram '):
                    continue
                row = dict(re.findall(r'(\w+)=(\d+)', line))
                phase = 'prefill' if int(row['prefill']) else 'decode'
                multiplicity, count = int(row['multiplicity']), int(row['experts'])
                triplets[phase][multiplicity] += count
                for tensor in inventory[model]['layers'][row['layer']]:
                    kind = tensor['type']
                    histograms[phase, kind][multiplicity] += count
                    weights[phase, kind][multiplicity] += count * tensor['expert_bytes']
            for (phase, kind), histogram in histograms.items():
                total = sum(histogram.values())
                byte_weights = weights[phase, kind]
                results.append(dict(trace=trace, policy=policy, phase=phase, quant=kind,
                                    histogram=dict(sorted(histogram.items())), unique_matrices=total,
                                    mean=sum(k * v for k, v in histogram.items()) / total,
                                    distinct_weight_bytes=sum(byte_weights.values()),
                                    routed_weight_bytes=sum(k * v for k, v in byte_weights.items()),
                                    byte_weighted_mean=sum(k * v for k, v in byte_weights.items()) / sum(byte_weights.values())))
            for phase, histogram in triplets.items():
                results.append(dict(trace=trace, policy=policy, phase=phase, quant='all-expert-triplets',
                                    histogram=dict(sorted(histogram.items())), unique_matrices=sum(histogram.values()),
                                    mean=sum(k * v for k, v in histogram.items()) / sum(histogram.values())))
    (OUT / 'multiplicity.json').write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(row['trace'], row['policy'], row['phase'], row['quant'], round(row['mean'], 4), row['histogram'])


if __name__ == '__main__':
    main()
