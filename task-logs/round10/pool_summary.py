#!/usr/bin/env python3
"""Apply the predeclared numeric, identity and throughput gates."""
import json
from pathlib import Path
import statistics

from summarize import summarize

OUT = Path(__file__).resolve().parent


def main():
    summary = summarize(OUT / 'pool-results', 'qwen-pool-control')
    events = [json.loads(x) for x in (OUT / 'pool-results/runs.jsonl').read_text().splitlines()]
    requests = [r for r in events if r['kind'] == 'request' and r['accepted']]
    table = {(r['config'], r['category']): r for r in summary['summary']}
    gains = {category: 100*(table['qwen-pool-native', category]['tok_s']/table['qwen-pool-control', category]['tok_s']-1)
             for category in ['code', 'english']}
    control = {(r['round'], r['prompt']): r for r in requests if r['config'] == 'qwen-pool-control'}
    candidates = [r for r in requests if r['config'] == 'qwen-pool-native']
    def identity(r):
        return (r['text_sha256'], r['tokens'], r['server_timings']['draft_n'], r['server_timings']['draft_n_accepted'])
    pairs_match = all(identity(r) == identity(control[r['round'], r['prompt']]) for r in candidates)
    repeats_match = all(len(r['text_hashes']) == 1 and len(set(tuple(c) for c in r['draft_counts'])) == 1
                        for r in summary['per_prompt'])
    quality = json.loads((OUT / 'pool-quality-verdict.json').read_text())
    complete = summary['complete'] and len(requests) == 48 and all(r['requests'] == 12 for r in table.values())
    cpu = {}
    for config in ['qwen-pool-control', 'qwen-pool-native']:
        cpu[config] = {category: statistics.median(r['engine_cpu_s_per_token'] for r in requests
            if r['config'] == config and r['prompt'].startswith('code-') == (category == 'code')) for category in ['code', 'english']}
    decision = dict(complete=complete, gain_pct=gains, all_paired_outputs_and_draft_counts_match=pairs_match,
        repetitions_stable=repeats_match, quality=quality, cpu_seconds_per_token=cpu,
        adopt=complete and pairs_match and repeats_match and quality['cache_off_bit_exact']
              and quality['change_pct'] <= .5 and all(g >= 3 for g in gains.values()),
        threshold='Both category medians at least +3%, all24 paired outputs/counts identical, numeric quality passed; no live installation authorized in this handoff.')
    (OUT / 'pool-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'pool-decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
