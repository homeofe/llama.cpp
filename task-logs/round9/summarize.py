#!/usr/bin/env python3
import collections
import json
from pathlib import Path
import statistics
import sys


def main():
    groups = collections.defaultdict(list)
    records = []
    for name in sys.argv[1:]:
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
    hashes = collections.defaultdict(dict)
    for row in records:
        hashes[row['config']][row['round'], row['prompt']] = row['text_sha256']
    comparisons = []
    for candidate in hashes:
        control = 'q-warm-draft' if candidate.startswith(('q-', 'qwen-')) else 'ling-cache-mtp3'
        if candidate == control or control not in hashes:
            continue
        common = hashes[candidate].keys() & hashes[control].keys()
        comparisons.append(dict(control=control, candidate=candidate, requests=len(common),
                                same_hash=sum(hashes[candidate][k] == hashes[control][k] for k in common)))
    print(json.dumps(dict(summary=summary, output_comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
