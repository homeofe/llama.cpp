#!/usr/bin/env python3
"""Summarize diagnostic traces without treating instrumented times as speed wins."""
import collections
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3
import statistics

OUT = Path(__file__).resolve().parent


def union_ns(intervals):
    end = total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def main():
    directory = OUT / 'q-warm-draft-profile'
    prefix = directory / 'q-warm-draft-3997100'
    requests = [row for line in (directory / 'runs.jsonl').read_text().splitlines()
                if (row := json.loads(line))['kind'] == 'request' and row['accepted']]
    intervals = [(r['epoch'] - r['wall_s'], r['epoch']) for r in requests]

    def active(t):
        return any(a <= t <= b for a, b in intervals)

    gpu = []
    with prefix.with_suffix('.gpu.csv').open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            stamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone(timedelta(hours=2))).timestamp()
            if active(stamp):
                gpu.append({k: float(v.split()[0]) for k, v in row.items() if k != 'timestamp'})
    cpus = [row for line in prefix.with_suffix('.cpu.jsonl').read_text().splitlines()
            if active((row := json.loads(line))['epoch'])]
    perf = prefix.with_suffix('.perf.txt').read_text()
    symbols = []
    for line in perf.splitlines():
        m = re.match(r'\s*([\d.]+)%\s+(\S+)\s+\[.\]\s+(.*)', line)
        if m:
            symbols.append(dict(cycles_pct=float(m[1]), library=m[2], symbol=re.split(r'\s{2,}',m[3])[0]))
    profile = dict(
        diagnostic_only=True, cuda_graphs=True, requests=len(requests),
        decode_wall_s=sum((r['tokens'] - 1) / r['decode_tok_s'] for r in requests),
        request_wall_s=sum(r['wall_s'] for r in requests),
        gpu_request_samples=len(gpu),
        gpu_request_mean={k: statistics.mean(row[k] for row in gpu) for k in gpu[0]},
        gpu_request_max={k: max(row[k] for row in gpu) for k in gpu[0]},
        cpu_request_samples=len(cpus),
        cpu_request_mean_busy_cores=statistics.mean(sum(row['busy_by_cpu'].values()) for row in cpus),
        cpu_request_median_reported_mhz=statistics.median(v for row in cpus for v in row['mhz_physical_cpus']),
        cpu_symbols=symbols[:20],
        cpu_backend_graph_times=[dict(tokens=int(t), calls=int(n), cpu_ms=float(ms)) for t, n, ms in
            re.findall(r'round9_cpu_profile tokens=(\d+) calls=(\d+) cpu_ms=([\d.]+)', (directory / 'q-warm-draft.server.log').read_text())],
        caveats=['CPU graph durations may overlap GPU work and include warmup.',
                 'OpenMP cycles include spin waiting; they are not useful math throughput.',
                 'GPU utilization is a time-activity sensor, not per-SM occupancy.'])
    (OUT / 'cpu-profile-summary.json').write_text(json.dumps(profile, indent=2) + '\n')
    connection = sqlite3.connect(OUT / 'gpu-nsys/trace.sqlite')
    kernels = connection.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id').fetchall()
    lo, hi = min(r[0] for r in kernels), max(r[1] for r in kernels)
    copies = connection.execute('select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?', (lo, hi)).fetchall()
    kinds = dict(connection.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
    aggregates = collections.defaultdict(lambda: [0, 0])
    for a, b, name in kernels:
        aggregates[name][0] += 1
        aggregates[name][1] += b - a
    result = dict(diagnostic_only=True, cuda_graphs=False, target_only=True, measured_tokens=80,
        boundary='first to last CUDA kernel, including four prefixes', window_s=(hi-lo)/1e9,
        kernels=len(kernels), kernel_union_ms=union_ns([(a,b) for a,b,_ in kernels])/1e6,
        copy_union_ms=union_ns([(a,b) for a,b,_,_ in copies])/1e6,
        kernel_or_copy_union_ms=union_ns([(a,b) for a,b,_ in kernels]+[(a,b) for a,b,_,_ in copies])/1e6,
        copies=[dict(kind=kinds.get(k,k), calls=sum(r[3]==k for r in copies),
                     bytes=sum(r[2] for r in copies if r[3]==k),
                     sum_ms=sum(r[1]-r[0] for r in copies if r[3]==k)/1e6) for k in sorted({r[3] for r in copies})],
        kernels_by_sum_time=[dict(name=name,calls=n,sum_ms=ns/1e6) for name,(n,ns) in
                             sorted(aggregates.items(), key=lambda item:item[1][1], reverse=True)],
        caveat='Graphs disabled; launch gaps and overlap differ from production. Never infer a production speedup from these times.')
    (OUT / 'nsys-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k!='kernels_by_sum_time'},indent=2))
    print(json.dumps({k:v for k,v in profile.items() if k not in ['cpu_symbols','cpu_backend_graph_times']},indent=2))


if __name__ == '__main__':
    main()
