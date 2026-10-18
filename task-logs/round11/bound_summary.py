#!/usr/bin/env python3
"""Summarize admission screening, preserving the cache-counter scope."""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'
spec = importlib.util.spec_from_file_location('summary10', ROOT / 'task-logs/round10/summarize.py')
assert spec and spec.loader
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def main():
    events = [json.loads(line) for line in (OUT / 'bound/runs.jsonl').read_text().splitlines()]
    stats = []
    for row in events:
        if row['kind'] != 'cache_pass' or not row['completed']:
            continue
        contexts = []
        context = None
        for line in row['cache_statistics']:
            match = re.search(r'context = (\d+) \((\w+)\)', line)
            if match:
                context = dict(id=match[1], kind=match[2])
                continue
            match = re.search(r'tokens = (\d+), hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%, uploads = (\d+), uploaded = ([\d.]+) MiB', line)
            if not match:
                continue
            assert context is not None
            context.update(tokens=int(match[1]), hits=int(match[2]), misses=int(match[3]), hit_rate_pct=float(match[4]), uploads=int(match[5]), uploaded_mib=float(match[6]))
            bursts = [re.search(r'context = (\d+), uploads = (\d+), MiB = ([\d.]+)', b) for b in row['prefill_bursts']]
            context['prefill_uploaded_mib'] = sum(float(b[3]) for b in bursts if b and b[1] == context['id'])
            context['decode_uploaded_mib'] = context['uploaded_mib'] - context['prefill_uploaded_mib']
            contexts.append(context)
            context = None
        rounds = row['speculative_rounds']
        stats.append(dict(config=row['config'], contexts=contexts, speculative_rounds=rounds,
            decode_uploaded_mib_per_round=sum(c['decode_uploaded_mib'] for c in contexts) / rounds if rounds else None))
    result = dict(complete=events[-1]['kind'] == 'all_complete', cache=stats,
        models={m: summary.summarize(OUT / 'bound', m + '-admit32') for m in ['qwen', 'ling']},
        scope='Speed: eight measured prompts. Cache counters and exact logged speculative-round count: those prompts plus the 16-token warmup. Hit rate includes prefill; upload rate subtracts all logged prompt bursts and excludes model initialization. All controls use identical level-5 logging.')
    (OUT / 'bound-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result['models']['qwen']['summary']:
        print(row['config'], row['category'], round(row['tok_s'], 3))
    for row in stats:
        print(row['config'], 'MiB/round', row['decode_uploaded_mib_per_round'], 'hits', [(c['kind'], c['hit_rate_pct']) for c in row['contexts']])


if __name__ == '__main__':
    main()
