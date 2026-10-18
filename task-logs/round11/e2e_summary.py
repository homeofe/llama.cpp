#!/usr/bin/env python3
"""Summarize accepted full passes, preserving per-round and same-output controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('summary10', ROOT / 'task-logs/round10/summarize.py')
summary10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen', 'ling'])
    args = parser.parse_args()
    directory = ROOT / 'task-logs/round11' / (args.model + '-e2e')
    control, candidate = args.model + '-control', args.model + '-async'
    result = summary10.summarize(directory, control)
    rows = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r.get('kind') == 'request' and r.get('accepted')]
    changes = []
    for category in ['code', 'english']:
        selected = [r for r in rows if r['prompt'].startswith('code-') == (category == 'code')]
        old = [r for r in selected if r['config'] == control]
        new = [r for r in selected if r['config'] == candidate]
        if not old or not new:
            continue
        per_round = []
        for rnd in sorted({r['round'] for r in selected}):
            a = [r['decode_tok_s'] for r in old if r['round'] == rnd]
            b = [r['decode_tok_s'] for r in new if r['round'] == rnd]
            if a and b:
                per_round.append(dict(round=rnd, control=statistics.median(a), candidate=statistics.median(b),
                    change_pct=(statistics.median(b) / statistics.median(a) - 1) * 100))
        index = {(r['round'], r['prompt']): r for r in old}
        equal = []
        same_text = 0
        for row in new:
            base = index.get((row['round'], row['prompt']))
            if base is None:
                continue
            text = row['text_sha256'] == base['text_sha256']
            same_text += text
            if text and all(row['server_timings'].get(k) == base['server_timings'].get(k)
                for k in ['draft_n', 'draft_n_accepted', 'predicted_n']):
                equal.append(dict(round=row['round'], prompt=row['prompt'],
                    change_pct=(row['decode_tok_s'] / base['decode_tok_s'] - 1) * 100))
        changes.append(dict(category=category, control_tok_s=statistics.median(r['decode_tok_s'] for r in old),
            candidate_tok_s=statistics.median(r['decode_tok_s'] for r in new),
            change_pct=(statistics.median(r['decode_tok_s'] for r in new) / statistics.median(r['decode_tok_s'] for r in old) - 1) * 100,
            per_round=per_round, same_text_pairs=same_text, exact_work_pairs=len(equal),
            exact_work_pair_changes=equal,
            exact_work_median_change_pct=statistics.median(r['change_pct'] for r in equal) if equal else None))
    result.update(changes=changes, all_48_requests=len(rows) == 48,
        speed_threshold_met=len(rows) == 48 and len(changes) == 2 and all(r['change_pct'] >= 3 for r in changes),
        caution='Speculative tok/s with changed text or draft counts is not a shape-identical kernel speedup; quality gates are separate.')
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'all_48_requests', 'rejected_passes', 'speed_threshold_met', 'changes']}, indent=2))


if __name__ == '__main__':
    main()
