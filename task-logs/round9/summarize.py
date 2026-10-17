#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--control')
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    records = []
    for name in args.files:
        for line in Path(name).read_text().splitlines():
            row = json.loads(line)
            if row.get('kind') == 'request' and row.get('accepted'):
                records.append(row)
                groups[row['config'], 'code' if row['prompt'].startswith('code-') else 'english'].append(row)
    summary = []
    for (config, category), rows in groups.items():
        rates = [r['decode_tok_s'] for r in rows]
        drafted = sum((r.get('server_timings') or {}).get('draft_n') or 0 for r in rows)
        accepted = sum((r.get('server_timings') or {}).get('draft_n_accepted') or 0 for r in rows)
        summary.append(dict(config=config, category=category, n=len(rows), median=statistics.median(rates),
                            minimum=min(rates), maximum=max(rates), acceptance=accepted / drafted if drafted else None,
                            vram_mib=max(int(r['vram_mib']) for r in rows), ttft=statistics.median(r['ttft_s'] for r in rows)))
        if all('engine_cpu_s' in r for r in rows):
            summary[-1]['cpu_s_per_token'] = sum(r['engine_cpu_s'] for r in rows) / sum(r['tokens'] for r in rows)
            summary[-1]['median_engine_busy_cores'] = statistics.median(r['engine_busy_cores'] for r in rows)
    hashes = collections.defaultdict(dict)
    for row in records:
        hashes[row['config']][row['round'], row['prompt']] = row
    comparisons = []
    for candidate in hashes:
        control = args.control or ('q-warm-draft' if candidate.startswith(('q-', 'qwen-')) else 'ling-cache-mtp3')
        if candidate == control or control not in hashes:
            continue
        common = hashes[candidate].keys() & hashes[control].keys()
        pairs = []
        for key in sorted(common):
            a, b = hashes[control][key], hashes[candidate][key]
            same_hash = a['text_sha256'] == b['text_sha256']
            same_work = same_hash and a['tokens'] == b['tokens'] and all(
                (a.get('server_timings') or {}).get(k) == (b.get('server_timings') or {}).get(k)
                for k in ['draft_n', 'draft_n_accepted', 'predicted_n'])
            pairs.append(dict(round=key[0], prompt=key[1], same_hash=same_hash, same_work=same_work,
                              gain_percent=(b['decode_tok_s'] / a['decode_tok_s'] - 1) * 100))
        comparisons.append(dict(control=control, candidate=candidate, requests=len(common),
                                same_hash=sum(p['same_hash'] for p in pairs),
                                same_work=sum(p['same_work'] for p in pairs), pairs=pairs))
    print(json.dumps(dict(summary=summary, output_comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
