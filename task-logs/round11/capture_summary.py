#!/usr/bin/env python3
"""Measure actual transfer/kernel overlap using clipped interval unions."""
import argparse
import collections
import json
from pathlib import Path
import sqlite3


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def duration(intervals):
    return sum(b - a for a, b in merge(intervals))


def overlap(left, right):
    a, b = merge(left), merge(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(directory):
    connection = sqlite3.connect('file:' + str(directory / 'trace.sqlite') + '?mode=ro', uri=True)
    strings = dict(connection.execute('select id,value from StringIds'))
    nvtx = [(a, b, text or strings.get(text_id, '')) for a, b, text, text_id in
        connection.execute('select start,end,text,textId from NVTX_EVENTS')]
    first = next(a for a, _, text in nvtx if text == 'first_token')
    end = next(b for _, b, text in nvtx if text == 'code-duration')
    pid = json.loads((directory / 'owned-pids.json').read_text())['server']
    global_pid = connection.execute('select globalPid from PROCESSES where pid=?', (pid,)).fetchone()[0]
    kernels = [(max(a, first), min(b, end), stream) for a, b, stream in connection.execute(
        'select start,end,streamId from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and end>? and start<?', (global_pid, first, end))]
    copies = [(max(a, first), min(b, end), stream, size, kind, source) for a, b, stream, size, kind, source in connection.execute(
        'select start,end,streamId,bytes,copyKind,srcKind from CUPTI_ACTIVITY_KIND_MEMCPY where globalPid=? and end>? and start<?', (global_pid, first, end))]
    kernel_streams = {stream for _, _, stream in kernels}
    compute = [(a, b) for a, b, _ in kernels]
    copy_kinds = dict(connection.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
    memory_kinds = dict(connection.execute('select id,label from ENUM_CUDA_MEM_KIND'))
    per_stream = []
    for stream in sorted({r[2] for r in copies}):
        selected = [r for r in copies if r[2] == stream]
        bulk = [r for r in selected if copy_kinds[r[4]] == 'Host-to-Device' and r[3] >= 102400]
        intervals = [(r[0], r[1]) for r in bulk]
        copied = duration(intervals)
        concurrent = overlap(intervals, compute)
        per_stream.append(dict(stream=stream, also_runs_kernels=stream in kernel_streams, all_copies=len(selected),
            bulk_htod_copies=len(bulk), bulk_htod_bytes=sum(r[3] for r in bulk), bulk_htod_union_ms=copied / 1e6,
            overlap_with_kernels_ms=concurrent / 1e6, overlap_pct=100 * concurrent / copied if copied else None,
            bulk_source_kinds=dict(collections.Counter(memory_kinds[r[5]] for r in bulk))))
    api_counts = collections.Counter(strings[name] for (name,) in connection.execute(
        'select nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid>=? and globalTid<? and end>? and start<?',
        (global_pid, global_pid + (1 << 24), first, end)))
    api_copies = collections.defaultdict(list)
    for start, finish, size, source, stream in connection.execute('''
        select r.start,r.end,m.bytes,m.srcKind,m.streamId
        from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_MEMCPY m
        on r.correlationId=m.correlationId
        where r.globalTid>=? and r.globalTid<? and m.globalPid=?
        and m.copyKind=1 and r.end>? and r.start<?''', (global_pid, global_pid + (1 << 24), global_pid, first, end)):
        group = ('scalar_map' if size == 4 else 'bulk' if size >= 102400 else 'small') + '/' + memory_kinds[source]
        api_copies[group].append((max(start, first), min(finish, end)))
    return dict(config=directory.name.removeprefix('capture-'), decode_span_ms=(end - first) / 1e6,
        kernel_union_ms=duration(compute) / 1e6, kernel_streams=sorted(kernel_streams), streams=per_stream,
        cuda_event_and_sync_calls={name: n for name, n in api_counts.items() if 'Event' in name or 'Synchronize' in name},
        host_htod_api={name: dict(calls=len(values), union_ms=duration(values) / 1e6,
            max_call_ms=max(b - a for a, b in values) / 1e6) for name, values in api_copies.items()},
        scope='One instrumented graph-off request; >=100 KiB host-to-device copies approximate expert uploads. Stream IDs and actual interval overlap prove scheduling, not uninstrumented end-to-end gain. CPU overlap is not included.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
