#!/usr/bin/env python3
"""Join paired visibility/control runs without adding overlapping clocks."""
import json
from pathlib import Path
import statistics
from typing import Any

from profile_logs import parse

ROOT = Path(__file__).resolve().parent


def phase_means(rows):
    names = list(dict.fromkeys(p['name'] for r in rows for p in r['phases']))
    return {name: {key: statistics.mean(p[key] for r in rows for p in r['phases'] if p['name'] == name)
                   for key in next(p for r in rows for p in r['phases'] if p['name'] == name) if key != 'name'}
            for name in names}


def direct_rounds(directory, log):
    boundary = json.loads((directory / 'request-boundaries.json').read_text())
    rows = []
    for r in log['rounds']:
        start = r['start_us']
        end = r.get('next_round_us', r.get('release_us', start))
        if start * 1000 < boundary['first_token_epoch_ns'] or end * 1000 > boundary['end_epoch_ns'] or end <= start:
            continue
        if len(r['candidate_us']) != 3 or len(r.get('verify_yield_ends_us', [])) != 2 or r.get('verify_tokens') != 4:
            continue
        points = [(start, 'draft_1_with_setup'),
                  (r['candidate_us'][0], 'draft_2'), (r['candidate_us'][1], 'draft_3'),
                  (r['candidate_us'][2], 'draft_tail'), (r['verify_start_us'], 'verify_target'),
                  (r['verify_yield_ends_us'][0], 'mtp_catchup'),
                  (r['verify_yield_ends_us'][1], 'sampling_accept_host'),
                  (r['accepted_us'], 'emit_and_host'), (end, 'end')]
        assert all(a <= b for (a, _), (b, _) in zip(points, points[1:]))
        rows.append(dict(wall_ms=(end-start)/1000, verify_tokens=r['verify_tokens'],
            phases=[dict(name=name, wall_ms=(b-a)/1000) for (a, name), (b, _) in zip(points, points[1:])]))
    return rows


def main():
    results: dict[str, dict[str, Any]] = {}
    for prompt in ['code-duration', 'en-1']:
        modes: dict[str, Any] = {}
        for mode in ['off', 'off-counters', 'on-counters', 'on-plain']:
            directory = ROOT / f'profile-{prompt}-{mode}'
            events = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
            request = next(x for x in events if x['kind'] == 'profile_request')
            assert request['accepted'] and events[-1]['kind'] == 'profile_exit' and events[-1]['exit'] == 0
            log = parse(directory)
            stats = log['request_statistics_delta']
            assert stats and stats['draft_tokens'] == request['server_timings']['draft_n']
            if mode == 'off':
                trace = json.loads((directory / 'summary.json').read_text())
                rows = [r for r in trace['rounds'] if r['full_three_step_draft'] and r['verify_tokens'] == 4]
                extra = {k: trace[k] for k in ['cpu_samples', 'cpu_leaf_modules', 'alignment_spread_ns', 'activity']}
            else:
                rows = direct_rounds(directory, log) if mode != 'on-plain' else []
                extra = {}
            assert all(r['verify_tokens'] == 4 for r in rows)
            counters = None
            if mode != 'on-plain':
                counter = json.loads((directory / 'cpu-counters.json').read_text())['decode_delta']
                counters = dict(calls=counter['calls'], ms_per_request_round=[v/1e6/stats['accept_calls'] for v in counter['cpu_ns']])
            modes[mode] = dict(request=request, stats=stats, complete_three_step_rounds=len(rows),
                mean_complete_round_ms=statistics.mean(r['wall_ms'] for r in rows) if rows else None,
                server_decode_ms=log['decode_times'][-1]['ms'],
                server_ms_per_request_round=log['decode_times'][-1]['ms']/stats['accept_calls'],
                phase_means=phase_means(rows) if rows else {}, cpu_graph_counters=counters, **extra)
        identity = {(r['request']['text_sha256'], r['request']['tokens'], r['stats']['draft_tokens'], r['stats']['accepted_tokens']) for r in modes.values()}
        assert len(identity) == 1, (prompt, identity)
        results[prompt] = modes
    (ROOT / 'profile-summary.json').write_text(json.dumps(dict(results=results,
        scope='One request per prompt/mode, diagnostic attribution, not a three-run performance claim. Full-round means omit first/last partial rounds. CPU graph clocks overlap GPU activity; bucket4 contains target verify plus MTP catch-up; sample percentages aggregate CPU activity across all threads.'), indent=2) + '\n')
    for prompt, modes in results.items():
        print(prompt)
        for mode, r in modes.items():
            print(mode, 'tok/s', r['request']['decode_tok_s'], 'ms/round', r['server_ms_per_request_round'], 'full', r['complete_three_step_rounds'])
            if r['phase_means']:
                print({k: round(v['wall_ms'], 3) for k, v in r['phase_means'].items()})
            if r['cpu_graph_counters']:
                print('cpu graph ms/round', [round(x, 3) for x in r['cpu_graph_counters']['ms_per_request_round']])


if __name__ == '__main__':
    main()
