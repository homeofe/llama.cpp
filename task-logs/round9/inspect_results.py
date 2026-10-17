#!/usr/bin/env python3
"""Read raw drafter timings or a completed Nsight SQLite capture."""
import argparse
import collections
import json
from pathlib import Path
import re
import sqlite3
import statistics
from typing import TypedDict


class DraftRequest(TypedDict):
    config: str
    prompt: str
    mean_length: float
    accepted: int
    drafted: int
    tok_s: float


def dspark(directory):
    rows = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
    assert rows[-1]['kind'] == 'all_complete'
    grouped = collections.defaultdict(list)
    for row in rows:
        if row['kind'] == 'request':
            grouped[row['config']].append(row)
    pairs: list[DraftRequest] = []
    for name, requests in grouped.items():
        log = (directory / (name + '.server.log')).read_text()
        lengths = re.findall(r'draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)', log)
        assert len(lengths) == len(requests) + 1, (name, 'unexpected restart or missing timing')
        for row, (accepted, drafted, length) in zip(requests, lengths[1:]):
            timing = row['server_timings']
            assert int(accepted) == timing['draft_n_accepted'] and int(drafted) == timing['draft_n']
            if row['accepted']:
                pairs.append(dict(config=name, prompt=row['prompt'], mean_length=float(length),
                                  accepted=int(accepted), drafted=int(drafted), tok_s=row['decode_tok_s']))
    summary = []
    for name in grouped:
        for code in [True, False]:
            selected = [r for r in pairs if r['config'] == name and r['prompt'].startswith('code-') == code]
            values = [r['mean_length'] for r in selected]
            summary.append(dict(config=name, category='code' if code else 'english',
                                n=len(selected), median_mean_length=statistics.median(values),
                                minimum=min(values), maximum=max(values)))
    return dict(summary=summary, requests=pairs,
                caveat='Server mean lengths include the bonus token and are rounded to two decimal places.')


def union(intervals):
    total = 0
    end = 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def nsys(path):
    db = sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True)
    kernels = db.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    lo, hi = min(a for a, _ in kernels), max(b for _, b in kernels)
    copies = [(max(lo, a), min(hi, b)) for a, b in db.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY')
              if a < hi and b > lo]
    runtime = [dict(name=n, calls=c, total_ms=t) for n, c, t in db.execute('''
        select s.value,count(*),sum(r.end-r.start)/1e6 from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on r.nameId=s.id where r.start>=? and r.end<=?
        group by s.value order by count(*) desc''', (lo, hi))]
    graph_calls = sum(r['calls'] for r in runtime if r['name'].startswith('cudaGraphLaunch'))
    return dict(source=str(path), diagnostic_only=True, captured_kernel_span_ms=(hi-lo)/1e6,
                captured_kernels=len(kernels), captured_kernel_union_ms=union(kernels)/1e6,
                captured_copy_union_ms=union(copies)/1e6,
                captured_combined_union_ms=union(kernels+copies)/1e6, graph_launches=graph_calls, runtime=runtime,
                kernel_timeline_complete=not bool(graph_calls),
                caveat='Instrumented target-only four-token windows; not end-to-end MTP performance. '
                       'With graph launches, default graph-level tracing omits individual graph nodes; '
                       'the captured kernel intervals cannot quantify total GPU activity or idle time.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['dspark', 'nsys'])
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    result = dspark(args.path) if args.kind == 'dspark' else nsys(args.path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
