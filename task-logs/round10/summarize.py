#!/usr/bin/env python3
"""Summarize complete, uncontaminated prompt passes without pooling retries."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def summarize(directory):
    events = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
    requests = [r for r in events if r.get('kind') == 'request' and r.get('accepted')]
    grouped = collections.defaultdict(list)
    for row in requests:
        grouped[row['config']].append(row)
    control = {(r['round'], r['prompt']): r for r in requests if r['config'] == 'ling-cache-mtp3'}
    summary = []
    pairs = []
    for name, rows in grouped.items():
        for category in ['code', 'english']:
            selected = [r for r in rows if r['prompt'].startswith('code-') == (category == 'code')]
            if not selected:
                continue
            rates = [r['decode_tok_s'] for r in selected]
            lengths = [r['mean_accepted_length_including_bonus'] for r in selected if 'mean_accepted_length_including_bonus' in r]
            drafted = sum(r['server_timings']['draft_n'] for r in selected)
            accepted = sum(r['server_timings']['draft_n_accepted'] for r in selected)
            matches = 0
            for row in selected:
                base = control.get((row['round'], row['prompt']))
                match = base is not None and row['text_sha256'] == base['text_sha256']
                matches += int(match)
                pairs.append(dict(config=name, category=category, round=row['round'], prompt=row['prompt'],
                    tok_s=row['decode_tok_s'], control_tok_s=base['decode_tok_s'] if base else None,
                    change_pct=(row['decode_tok_s']/base['decode_tok_s']-1)*100 if base else None,
                    same_text=match, text_sha256=row['text_sha256'],
                    control_sha256=base['text_sha256'] if base else None,
                    draft_n=row['server_timings']['draft_n'], draft_n_accepted=row['server_timings']['draft_n_accepted'],
                    mean_length=row.get('mean_accepted_length_including_bonus')))
            summary.append(dict(config=name, category=category, requests=len(selected),
                tok_s=statistics.median(rates), minimum=min(rates), maximum=max(rates),
                acceptance_pct=100*accepted/drafted if drafted else None,
                median_mean_length=statistics.median(lengths) if lengths else None,
                measured_lengths=len(lengths), text_matches_control=matches,
                max_foreign_cores=max(r['foreign_cores'] for r in selected),
                peak_vram_mib=max(int(r['vram_mib']) for r in selected)))
    return dict(complete=events[-1]['kind']=='all_complete', summary=summary, requests=pairs,
        rejected_passes=sum(r.get('kind')=='requeue' for r in events),
        length_definition='Server mean includes bonus token; rounded to 2 decimal places.',
        hashes_definition='SHA256 of all streamed text, not the truncated preview.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
