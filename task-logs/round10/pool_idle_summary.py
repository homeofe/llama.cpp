#!/usr/bin/env python3
"""Report the bounded zero-polling screen separately from confirmed results."""
import json
from pathlib import Path
import statistics

from summarize import summarize

OUT = Path(__file__).resolve().parent


def main():
    summary = summarize(OUT / 'pool-idle-results', 'qwen-pool-control')
    events = [json.loads(line) for line in (OUT / 'pool-idle-results/runs.jsonl').read_text().splitlines()]
    rows = [r for r in events if r['kind'] == 'request' and r['accepted']]
    assert summary['complete'] and len(rows) == 16
    table = {(r['config'], r['category']): r for r in summary['summary']}
    cpu = {}
    for name in ['qwen-pool-control', 'qwen-pool-native-idle']:
        cpu[name] = {category: statistics.median(r['engine_cpu_s_per_token'] for r in rows
            if r['config'] == name and r['prompt'].startswith('code-') == (category == 'code'))
            for category in ['code', 'english']}
    gains = {cat: 100*(table['qwen-pool-native-idle', cat]['tok_s']/table['qwen-pool-control', cat]['tok_s']-1)
             for cat in ['code', 'english']}
    cpu_change = {cat: 100*(cpu['qwen-pool-native-idle'][cat]/cpu['qwen-pool-control'][cat]-1)
                  for cat in ['code', 'english']}
    base = {r['prompt']: r for r in rows if r['config'] == 'qwen-pool-control'}
    def identity(r):
        return r['text_sha256'], r['tokens'], r['server_timings']['draft_n'], r['server_timings']['draft_n_accepted']
    same = all(identity(r) == identity(base[r['prompt']]) for r in rows if r['config'] == 'qwen-pool-native-idle')
    result = dict(screen_only=True, requests=16, rounds=1, gain_pct=gains, cpu_seconds_per_token=cpu,
        cpu_change_pct=cpu_change, paired_outputs_and_counts_match=same,
        qualifies_for_confirmation=same and all(g >= 3 for g in gains.values()), adopted=False,
        scope='One fresh interleaved pair. CPU work includes prefill; no electrical energy was measured. A screen alone does not qualify a configuration for adoption.')
    (OUT / 'pool-idle-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'pool-idle-decision.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
