#!/usr/bin/env python3
"""Keep numerical, runtime and quiet-machine evidence separate."""
import filecmp
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gate(directory):
    rows = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
    result = next(r for r in reversed(rows) if r['kind'] == 'quality')
    assert result['accepted'], 'quality run did not pass its gate: ' + str(directory)
    return result


def main():
    result = {}
    for model in ['qwen', 'ling']:
        original = next((ROOT / 'task-logs/round9' / (model + '-off-probe')).glob('*.logits.bin'))
        directory = OUT / (model + '-off-async-probe')
        candidate = next(directory.glob('*.logits.bin'))
        probe = dict(control=str(original.relative_to(ROOT)), candidate=str(candidate.relative_to(ROOT)),
            bytes=candidate.stat().st_size, bit_exact=filecmp.cmp(original, candidate, shallow=False),
            control_sha256=digest(original), candidate_sha256=digest(candidate), gate=gate(directory))
        ppl = []
        names = [model + '-off-async-ppl', model + ('-r7' if model == 'qwen' else '-r7-4608') + '-async-ppl']
        for name in names:
            directory = OUT / name
            accepted = gate(directory)
            log = directory / ('attempt-' + str(accepted['attempt']) + '.log')
            values = re.findall(r'Final estimate: PPL = ([\d.]+)', log.read_text())
            assert values, 'perplexity value missing'
            ppl.append(dict(name=name, ppl=float(values[-1]), log=str(log.relative_to(ROOT)), gate=accepted))
        for row in ppl:
            row.update(change_pct=(row['ppl'] / ppl[0]['ppl'] - 1) * 100, numerical_gate=row['ppl'] <= 1.005 * ppl[0]['ppl'])
        result[model] = dict(probe=probe, perplexity=ppl, quality_passed=probe['bit_exact'] and all(r['numerical_gate'] for r in ppl))
    result['scope'] = 'Cache off: full-vocabulary 32-token probes over four topics, window 4, prefix 32. PPL: test-de, 12 chunks, context 512, ubatch 4. These checks exercise target/cache behavior; drafter behavior is covered separately by output hashes and panels.'
    (OUT / 'quality-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({m: dict(bit_exact=result[m]['probe']['bit_exact'], quality_passed=result[m]['quality_passed'],
        ppl=[r['ppl'] for r in result[m]['perplexity']]) for m in ['qwen', 'ling']}, indent=2))


if __name__ == '__main__':
    main()
