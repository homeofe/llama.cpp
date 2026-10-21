#!/usr/bin/env python3
"""Measure causal previous-window routing prediction, with no cache-hit claim."""
import collections
import hashlib
import json
from pathlib import Path

TRACE = Path('/opt/llm-gateway/data/moe-quant/traces/qwen38-bench8.jsonl')
OUT = Path(__file__).resolve().parent / 'routing-bound.json'


def replay(shift=0, oracle=False):
    previous = {}
    stats = collections.Counter()
    per_request = []
    request = collections.Counter()
    for line in TRACE.open():
        record = json.loads(line)
        if record.get('draft') or not record.get('layers'):
            continue
        if record.get('prefill') or record['positions'][0] == 0:
            if request:
                per_request.append(dict(request))
                request.clear()
            previous.clear()
            continue
        if not any(layer['cache'] for layer in record['layers']):
            continue
        for layer in record['layers']:
            actual = {expert for row in layer['ids'] for expert in row}
            predicted = previous.get(layer['layer'], set())
            if predicted:
                predicted = actual if oracle else {(expert + shift) % layer['n_expert'] for expert in predicted}
                hits = len(actual & predicted)
                counts = dict(pairs=1, actual=len(actual), predicted=len(predicted), correct=hits,
                              actual_bytes=len(actual)*layer['expert_bytes'],
                              predicted_bytes=len(predicted)*layer['expert_bytes'],
                              correct_bytes=hits*layer['expert_bytes'])
                stats.update(counts)
                request.update(counts)
            previous[layer['layer']] = actual
    if request:
        per_request.append(dict(request))
    return dict(stats), per_request


def main():
    counts, per_request = replay()
    oracle, _ = replay(oracle=True)
    shifted, _ = replay(shift=127)
    assert oracle['correct'] == oracle['predicted'] == oracle['actual']
    assert shifted['correct'] < counts['correct'] / 2
    report = dict(trace=str(TRACE), trace_sha256=hashlib.sha256(TRACE.read_bytes()).hexdigest(),
                  predictor='Previous target verify window union, same layer; reset on prefill/request boundary',
                  counts=counts, per_request=per_request,
                  recall=counts['correct']/counts['actual'],
                  precision=counts['correct']/counts['predicted'],
                  byte_precision=counts['correct_bytes']/counts['predicted_bytes'],
                  oracle_sanity=oracle, shifted_id_negative_control=shifted,
                  limitations=['Cache residency is not present in this trace; cache=true means use_cache, not hit.',
                               'Counts include resident and nonresident experts; no miss-only speedup is inferred.',
                               'Old n3 trace, not a benchmark of a routing implementation.',
                               'Expert IDs do not provide the activation inputs needed to compute future layers.'])
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('recall', 'precision', 'byte_precision')}))


if __name__ == '__main__':
    main()
