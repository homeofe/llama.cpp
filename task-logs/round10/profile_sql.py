#!/usr/bin/env python3
"""Summarize captured activity as interval unions; overlapping timers remain separate."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import re
import sqlite3
import statistics

from profile_logs import parse


def union_ns(intervals):
    total = 0
    previous = None
    for start, end in sorted(intervals):
        if previous is None or start > previous:
            total += end - start
        elif end > previous:
            total += end - previous
        previous = max(previous or end, end)
    return total


class Intervals:
    def __init__(self, intervals):
        self.values = sorted(intervals)
        self.starts = [a for a, _ in self.values]
        self.ends = []
        maximum = 0
        for _, end in self.values:
            maximum = max(maximum, end)
            self.ends.append(maximum)

    def clip(self, start, end):
        lower = bisect.bisect_right(self.ends, start)
        upper = bisect.bisect_left(self.starts, end)
        return [(max(a, start), min(b, end)) for a, b in self.values[lower:upper] if b > start]


def summarize(directory):
    connection = sqlite3.connect('file:' + str(directory / 'trace.sqlite') + '?mode=ro', uri=True)
    strings = dict(connection.execute('select id,value from StringIds'))
    nvtx = [(a, b, text or strings.get(text_id, '')) for a, b, text, text_id in
            connection.execute('select start,end,text,textId from NVTX_EVENTS')]
    first = next(a for a, _, text in nvtx if text == 'first_token')
    request = next((a, b, text) for a, b, text in nvtx if text in ['code-duration', 'en-1'])
    end = request[1]
    offsets = [a - int(text.split(';')[0].split('=')[1]) for a, _, text in nvtx if text.startswith('epoch_ns=')]
    offset = statistics.median(offsets)
    pid = json.loads((directory / 'owned-pids.json').read_text())['server']
    global_pid = connection.execute('select globalPid from PROCESSES where pid=?', (pid,)).fetchone()[0]
    kernels = [(a, b, strings[name]) for a, b, name in connection.execute(
        'select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and end>? and start<?', (global_pid, first, end))]
    copies = list(connection.execute('select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where globalPid=? and end>? and start<?', (global_pid, first, end)))
    apis = [(a, b, strings[name]) for a, b, name in connection.execute(
        'select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid>=? and globalTid<? and end>? and start<?',
        (global_pid, global_pid + (1 << 24), first, end))]
    groups = collections.defaultdict(list)
    kernel_counts = collections.Counter()
    kernel_time = collections.Counter()
    for a, b, name in kernels:
        group = 'expert_masked' if '_moe<' in name and re.search(r'\(bool\)1>', name) else 'expert_unmasked' if '_moe<' in name else 'dense_and_other'
        groups[group].append((a, b))
        kernel_counts[name] += 1
        kernel_time[name] += max(0, min(b, end) - max(a, first))
    groups['kernels'] = [(a, b) for a, b, _ in kernels]
    groups['copies'] = [(a, b) for a, b, _, _ in copies]
    groups['gpu_activity'] = groups['kernels'] + groups['copies']
    groups['host_cuda_sync'] = [(a, b) for a, b, name in apis if 'Synchronize' in name]
    groups['host_cuda_api'] = [(a, b) for a, b, _ in apis]
    indexed = {key: Intervals(values) for key, values in groups.items()}

    log = parse(directory)
    rounds = []
    for row in log['rounds']:
        start_ns = row['start_us'] * 1000 + offset
        finish_ns = (row.get('next_round_us') or row.get('release_us') or row['start_us']) * 1000 + offset
        if start_ns < first or finish_ns > end or finish_ns <= start_ns:
            continue
        phases = []
        points = [(start_ns, 'draft_1_with_setup')]
        for position in range(3):
            if position in row['candidate_us']:
                points.append((row['candidate_us'][position] * 1000 + offset, ['draft_2', 'draft_3', 'draft_tail'][position]))
        if 'verify_start_us' in row:
            yields = row.get('verify_yield_ends_us', [])
            if len(yields) == 2:
                points.append((row['verify_start_us'] * 1000 + offset, 'verify_target'))
                points.append((yields[0] * 1000 + offset, 'mtp_catchup'))
                points.append((yields[1] * 1000 + offset, 'sampling_accept_host'))
            else:
                points.append((row['verify_start_us'] * 1000 + offset, 'verify_catchup_and_sample'))
        if 'accepted_us' in row:
            points.append((row['accepted_us'] * 1000 + offset, 'emit_and_host'))
        points.append((finish_ns, 'end'))
        full_draft = len(row['candidate_us']) == 3 and 'draft_end_us' in row
        for (a, name), (b, _) in zip(points, points[1:]):
            if b < a:
                raise ValueError('non-monotonic server boundaries')
            phases.append(dict(name=name, wall_ms=(b-a)/1e6,
                **{key + '_union_ms': union_ns(values.clip(a, b))/1e6 for key, values in indexed.items()}))
        rounds.append(dict(start_ns=start_ns, end_ns=finish_ns, full_three_step_draft=full_draft,
            verify_tokens=row.get('verify_tokens'), verify_yield_boundaries=len(row.get('verify_yield_ends_us', [])),
            wall_ms=(finish_ns-start_ns)/1e6, phases=phases))

    leaves = connection.execute('''select s.value,m.value,c.originalIP,c.unresolved,count(*)
        from COMPOSITE_EVENTS e join SAMPLING_CALLCHAINS c on c.id=e.id
        join StringIds s on s.id=c.symbol join StringIds m on m.id=c.module
        where e.globalTid>=? and e.globalTid<? and e.start>=? and e.start<=? and c.stackDepth=0
        group by s.value,m.value,c.originalIP,c.unresolved order by count(*) desc''', (global_pid, global_pid+(1 << 24), first, end)).fetchall()
    sample_count = connection.execute('select count(*) from COMPOSITE_EVENTS where globalTid>=? and globalTid<? and start>=? and start<=?',
        (global_pid, global_pid+(1 << 24), first, end)).fetchone()[0]
    module_counts = collections.Counter()
    symbol_counts = collections.Counter()
    for symbol, module, _, _, count in leaves:
        module_counts[module] += count
        symbol_counts[(module, symbol)] += count
    api_counts = collections.Counter(name for _, _, name in apis)
    copy_bytes = collections.Counter()
    copy_intervals = collections.defaultdict(list)
    for a, b, kind, size in copies:
        copy_bytes[kind] += size
        copy_intervals[kind].append((max(a, first), min(b, end)))
    return dict(request=request[2], first_token_ns=first, end_ns=end, decode_span_ms=(end-first)/1e6,
        server_pid=pid, epoch_to_trace_ns=offset, alignment_spread_ns=max(offsets)-min(offsets),
        activity={key+'_union_ms': union_ns(value.clip(first, end))/1e6 for key, value in indexed.items()},
        cuda_api_counts=dict(api_counts), copy_bytes_by_kind=dict(copy_bytes),
        copy_union_ms_by_kind={kind: union_ns(values)/1e6 for kind, values in copy_intervals.items()},
        kernels=[dict(name=name,count=kernel_counts[name],sum_ms=duration/1e6) for name,duration in kernel_time.most_common()],
        cpu_samples=sample_count, cpu_leaf_modules=dict(module_counts),
        cpu_leaf_symbols=[dict(module=m,symbol=s,count=n) for (m,s),n in symbol_counts.most_common()],
        cpu_leaf_ips=[dict(symbol=s,module=m,ip=ip,unresolved=u,count=n) for s,m,ip,u,n in leaves],
        server_logs=log, rounds=rounds,
        scope='CPU sample shares aggregate activity across threads; they are not wall-time shares. CUDA API, kernel and copy unions overlap each other and CPU graphs. Graph-enabled traces omit graph nodes in this Nsight version; their GPU union is incomplete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
