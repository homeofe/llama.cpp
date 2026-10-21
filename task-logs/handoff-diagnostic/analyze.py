#!/usr/bin/env python3
"""Correlate host ranges and GPU activity, preserving limits on causality."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import re
import sqlite3
import statistics


class Intervals(list):
    def __init__(self, values):
        super().__init__(values)
        self.ends = [b for a, b in values]
        self.prefix = [0]
        for a, b in values:
            self.prefix.append(self.prefix[-1] + b-a)

    def before(self, t):
        i = bisect.bisect_right(self.ends, t)
        return self.prefix[i] + (max(0, t-self[i][0]) if i < len(self) else 0)


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return Intervals(result)


def overlap(intervals, low, high):
    indexed = intervals if isinstance(intervals, Intervals) else Intervals(intervals)
    return indexed.before(high)-indexed.before(low)


def describe(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    return {'n': len(values), 'sum_ms': sum(values) / 1e6, 'median_us': statistics.median(values) / 1e3,
            'p95_us': values[min(len(values) - 1, int(.95 * len(values)))] / 1e3,
            'min_us': values[0] / 1e3, 'max_us': values[-1] / 1e3}


def analyze(out):
    db = sqlite3.connect((out / 'trace.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings = dict(db.execute('select id,value from StringIds'))
    def rows(table):
        return [dict(r) for r in db.execute('select * from ' + table)] if table in tables else []
    marks = rows('NVTX_EVENTS')
    for r in marks:
        r['label'] = r['text'] or strings.get(r['textId'], '')
    request = [r for r in marks if r['label'] == 'handoff-request' and r['end']]
    assert len(request) == 1, 'one complete request range required'
    lo, hi = request[0]['start'], request[0]['end']
    kernels = rows('CUPTI_ACTIVITY_KIND_KERNEL')
    for r in kernels:
        r['name'] = strings[r['demangledName']]
    copies = rows('CUPTI_ACTIVITY_KIND_MEMCPY')
    activity = kernels + copies + rows('CUPTI_ACTIVITY_KIND_MEMSET')
    productive = union((r['start'], r['end']) for r in activity if not re.search(r'wait_flag_ge(?:_or)?_kernel|doorbell_wait_kernel', r.get('name', '')))
    busy = union((r['start'], r['end']) for r in activity)
    events = [json.loads(s) for s in (out / 'runs.jsonl').read_text().splitlines()]
    result = next(e for e in events if e['kind'] == 'profiled_request')
    summary = {'benchmark_valid': False, 'quiet': result['quiet'], 'request': result,
               'window_ms': (hi-lo)/1e6, 'productive_union_ms': overlap(productive, lo, hi)/1e6,
               'wait_only_ms': (overlap(busy, lo, hi)-overlap(productive, lo, hi))/1e6,
               'device_idle_ms': (hi-lo-overlap(busy, lo, hi))/1e6}
    intervals = []
    transitions = []
    if (out / 'bpf.txt').exists():
        brackets = dict(json.loads((out / 'clocks.json').read_text()))
        anchors = []
        for m in marks:
            if m['label'].startswith('handoff-clock:'):
                before = int(m['label'].split(':')[1])
                after = brackets[before]
                anchors.append({'mono': before, 'width_ns': after-before,
                                'offset_mid_ns': m['start'] - (before+after)/2})
        assert len(anchors) == 4, 'cross-clock markers missing'
        offset_low = max(a['offset_mid_ns']-a['width_ns']/2 for a in anchors)
        offset_high = min(a['offset_mid_ns']+a['width_ns']/2 for a in anchors)
        assert offset_low <= offset_high, 'clock brackets do not support a constant offset'
        offset = (offset_low+offset_high)/2
        uncertainty = (offset_high-offset_low)/2
        summary['clock_alignment'] = {'offset_ns': offset, 'observed_envelope_ns': uncertainty, 'anchors': anchors,
                                      'note': 'Intersection of all four clock brackets, including the request end; assumes a constant offset. Not a bound on uprobe overhead or unobserved clock drift.'}
        stacks = collections.defaultdict(list)
        phases = []
        for line in (out / 'bpf.txt').read_text().splitlines():
            if not re.match(r'^[BE],', line):
                continue
            edge, kind, stamp, tid, arg1, arg2 = line.split(',')
            t = int(stamp) + offset
            key = (kind, tid)
            if edge == 'B':
                stacks[key].append((t, int(arg1), int(arg2)))
            else:
                assert stacks[key], 'unmatched return probe'
                start, arg1, arg2 = stacks[key].pop()
                if lo <= start < t <= hi:
                    row = {'start': start, 'end': t, 'tid': int(tid), 'arg1': arg1, 'arg2': arg2}
                    (intervals if kind == 'dispatch' else phases).append(row)
        assert not any(stacks.values()), 'unfinished host range'
        intervals.sort(key=lambda r:r['start'])
        doors = sorted((r for r in kernels if 'doorbell_publish' in r['name'] and lo <= r['start'] < hi), key=lambda r:r['start'])
        consumers = sorted((r for r in kernels if 'copy_rows_from_mapped_kernel' in r['name'] and lo <= r['start'] < hi), key=lambda r:r['start'])
        summary['coverage'] = {'dispatch': len(intervals), 'doorbells': len(doors), 'consumers': len(consumers), 'phases': len(phases)}
        assert intervals and len(intervals) == len(doors) == len(consumers), 'cannot establish one-to-one handoffs'
        for host, door, consume in zip(intervals, doors, consumers):
            assert 0 <= host['arg2'] < 48 and 1 <= host['arg1'] <= 8, 'native argument ABI mismatch'
            assert door['start'] - uncertainty <= host['start'] <= consume['end'] + uncertainty, 'event order mismatch'
            phases_in = [p for p in phases if host['start'] <= p['start'] and p['end'] <= host['end'] and p['tid'] == host['tid']]
            transitions.append({'tokens': host['arg1'], 'layer': host['arg2'], 'cpu_ns':host['end']-host['start'],
                                'after_publish_kernel_ns': host['start']-door['end'],
                                'before_consume_kernel_ns': consume['start']-host['end'],
                                'pool_phase_ns': sum(p['end']-p['start'] for p in phases_in)})
        summary['transition_definition'] = 'Same-order one-to-one doorbell / dispatch / mapped-result-copy. Negative deltas may reflect publish-before-kernel-end, clock envelope or uprobe overhead; do not clamp them into a saving.'
        summary['phases'] = {str(k): describe([r['end']-r['start'] for r in phases if r['arg1']==k]) for k in sorted({r['arg1'] for r in phases})}
    else:
        for m in marks:
            match = re.search(r'^handoff-cpu tokens=(\d+) node=(\S+) weight=(\S+)', m['label'])
            if match and m['end'] and lo <= m['start'] < m['end'] <= hi:
                tokens, node, weight = int(match[1]), match[2], match[3]
                layer_match = re.search(r'blk\.(\d+)\.', weight) or re.search(r'-(\d+)$', node)
                if tokens and layer_match and int(layer_match[1]) < 48:
                    intervals.append({'start': m['start'], 'end': m['end'], 'tokens': tokens, 'layer': int(layer_match[1]), 'label':m['label']})
        intervals.sort(key=lambda r:r['start'])
        d2h = sorted((r for r in copies if r['copyKind']==2), key=lambda r:r['end'])
        h2d = sorted((r for r in copies if r['copyKind']==1), key=lambda r:r['start'])
        dend = [r['end'] for r in d2h]; hstart = [r['start'] for r in h2d]
        for i, host in enumerate(intervals):
            row = {'tokens':host['tokens'], 'layer':host['layer'], 'cpu_ns':host['end']-host['start']}
            j = bisect.bisect_right(dend, host['start'])-1
            previous = intervals[i-1]['end'] if i else lo
            if j >= 0 and d2h[j]['start'] >= previous:
                row['after_last_d2h_ns'] = host['start']-d2h[j]['end']
            j = bisect.bisect_left(hstart, host['end'])
            following = intervals[i+1]['start'] if i+1<len(intervals) else hi
            if j < len(h2d) and h2d[j]['end'] <= following:
                row['before_first_h2d_ns'] = h2d[j]['start']-host['end']
            transitions.append(row)
        summary['coverage'] = {'cpu_expert_ranges':len(intervals), 'all_cpu_ranges':sum(m['label'].startswith('handoff-cpu ') for m in marks)}
        summary['transition_definition'] = 'Nearest DtoH before CPU / HtoD after CPU within adjacent CPU boundaries. Temporal association only: without tensor/stream ownership this is not a proven data dependency or removable latency.'
    assert intervals, 'host instrumentation produced no usable expert ranges'
    for r in intervals:
        r['productive_overlap_ns'] = overlap(productive, r['start'], r['end'])
    summary['cpu_ranges'] = describe([r['end']-r['start'] for r in intervals])
    summary['cpu_without_productive_gpu'] = describe([r['end']-r['start']-r['productive_overlap_ns'] for r in intervals])
    summary['by_tokens'] = {}
    for n in sorted({r['tokens'] for r in transitions}):
        rs = [r for r in transitions if r['tokens']==n]
        fields = sorted(set().union(*(r.keys() for r in rs))-{'tokens','layer'})
        summary['by_tokens'][str(n)] = {f:describe([r[f] for r in rs if f in r]) for f in fields}
    (out / 'host-ranges.json').write_text(json.dumps(intervals) + '\n')
    (out / 'transitions.json').write_text(json.dumps(transitions) + '\n')
    (out / 'handoff-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    analyze(ap.parse_args().out)
