#!/usr/bin/env python3
"""Summarize accepted quality runs and validate the full logit dump sizes."""
import hashlib
import json
from pathlib import Path
import re
from typing import TypedDict

ROOT = Path(__file__).resolve().parent


class PerplexityResult(TypedDict, total=False):
    name: str
    ppl: float
    stderr: float
    log: str
    relative_percent: float
    within_half_percent: bool


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    rows: list[PerplexityResult] = []
    for directory in sorted(ROOT.glob('*-ppl')):
        marker = directory / 'accepted.txt'
        if not marker.exists():
            continue
        log = (directory / marker.read_text().strip()).read_text()
        match = re.search(r'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)', log)
        if not match:
            raise RuntimeError(f'{directory}: no final PPL')
        rows.append(dict(name=directory.name, ppl=float(match[1]), stderr=float(match[2]),
                         log=str(directory / marker.read_text().strip())))
    baselines = {r['name'].split('-')[0]: r['ppl'] for r in rows if r['name'].endswith('-off-ppl')}
    for row in rows:
        baseline = baselines.get(row['name'].split('-')[0])
        if baseline is not None:
            row['relative_percent'] = (row['ppl'] / baseline - 1) * 100
            row['within_half_percent'] = row['ppl'] <= baseline * 1.005
    probes = []
    for model in ['qwen', 'ling']:
        pair = []
        for mode in ['off', 'off-knobs']:
            directory = ROOT / f'{model}-{mode}-probe'
            if not (directory / 'accepted.txt').exists():
                break
            report = json.loads((directory / 'logits.report.json').read_text())
            assert len(report['topics']) == 4 and report['window_tokens'] == 4
            assert report['prefix_tokens'] == 32
            positions = sum(len(t['tokens']) for t in report['topics'])
            assert positions == 128
            path = directory / 'logits.logits.bin'
            expected = positions * report['n_vocab'] * 4
            if path.stat().st_size != expected:
                raise RuntimeError(f'{path}: incorrect logit size')
            pair.append(dict(mode=mode, bytes=expected, positions=positions, sha256=digest(path)))
        if len(pair) == 2:
            probes.append(dict(model=model, bit_exact=pair[0]['sha256'] == pair[1]['sha256'], runs=pair))
    result = dict(perplexity=rows, probes=probes)
    (ROOT / 'quality-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
