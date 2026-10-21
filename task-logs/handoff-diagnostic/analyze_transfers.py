#!/usr/bin/env python3
"""Expand the aligned CPU boundary through event readiness and GPU reduction."""
import bisect
import collections
import json
from pathlib import Path
import sqlite3
from analyze import describe, overlap, union

out = Path(__file__).resolve().parent / 'aligned'
c = sqlite3.connect((out / 'trace.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
c.row_factory = sqlite3.Row
strings = dict(c.execute('select id,value from StringIds'))
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
def rows(table):
    return [dict(r) for r in c.execute('select * from ' + table)] if table in tables else []
runtime = rows('CUPTI_ACTIVITY_KIND_RUNTIME')
apis = {r['correlationId']:r for r in runtime}
for r in runtime:
    r['name'] = strings[r['nameId']]
kernels = rows('CUPTI_ACTIVITY_KIND_KERNEL')
for r in kernels:
    r['name'] = strings[r['demangledName']]
copies = rows('CUPTI_ACTIVITY_KIND_MEMCPY')
activity = kernels+copies+rows('CUPTI_ACTIVITY_KIND_MEMSET')
busy = union((r['start'],r['end']) for r in activity)
parts = collections.defaultdict(list)
for r in activity:
    parts[(r['contextId'],r['streamId'],r['correlationId'])].append(r)
streams = collections.defaultdict(list)
for (context,stream,corr), group in parts.items():
    if corr in apis:
        streams[(context,stream)].append((apis[corr]['start'],max(r['end'] for r in group),corr))
for key in streams:
    streams[key].sort()
records = collections.defaultdict(list)
for r in rows('CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
    if r['correlationId'] in apis:
        r['host_start'] = apis[r['correlationId']]['start']
        records[(r['contextId'],r['eventId'])].append(r)
for key in records:
    records[key].sort(key=lambda r:r['host_start'])
syncs = sorted((r for r in rows('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION') if r['syncType']==2),key=lambda r:r['end'])
sync_ends = [r['end'] for r in syncs]
reductions = sorted((r for r in kernels if r['name'].startswith('moe_weighted_reduction_f32')),key=lambda r:r['start'])
reduce_starts = [r['start'] for r in reductions]
host = json.loads((out / 'host-ranges.json').read_text())
api_groups = {}
for category, predicate in {
    'graph_lifecycle': lambda n:any(x in n for x in ('GraphInstantiate','GraphExecUpdate','GraphDestroy','GraphExecDestroy','StreamBeginCapture','StreamEndCapture')),
    'graph_launch':lambda n:'GraphLaunch' in n,
    'capture_kernel_launch':lambda n:'LaunchKernel' in n,
    'copy_api':lambda n:'Memcpy' in n,
    'synchronize_api':lambda n:'Synchronize' in n,
}.items():
    api_groups[category] = union((r['start'],r['end']) for r in runtime if predicate(r['name']))
all_apis = union((r['start'],r['end']) for r in runtime)
detail = []
for i,h in enumerate(host):
    row = {'tokens':h['tokens'],'layer':h['layer'],'cpu_start':h['start'],'cpu_end':h['end']}
    prev = host[i-1]['end'] if i else 0
    j = bisect.bisect_right(sync_ends,h['start'])-1
    if j>=0 and syncs[j]['start']>=prev:
        sync = syncs[j]
        candidates = records[(sync['contextId'],sync['eventId'])]
        times = [r['host_start'] for r in candidates]
        k = bisect.bisect_right(times,sync['start'])-1
        if k>=0:
            record = candidates[k]
            issued = streams[(record['contextId'],record['streamId'])]
            k = bisect.bisect_left(issued,(record['host_start'],))-1
            if k>=0:
                ready = issued[k][1]
                if prev <= ready <= h['start']:
                    row.update(input_ready=ready,input_elapsed_ns=h['start']-ready,
                               input_gpu_idle_ns=h['start']-ready-overlap(busy,ready,h['start']))
    j = bisect.bisect_left(reduce_starts,h['end'])
    following = host[i+1]['start'] if i+1<len(host) else float('inf')
    if j<len(reductions) and reductions[j]['start']<following:
        consume = reductions[j]['start']
        row.update(reduction_start=consume,return_elapsed_ns=consume-h['end'],
                   return_gpu_idle_ns=consume-h['end']-overlap(busy,h['end'],consume),
                   return_outside_cuda_api_ns=consume-h['end']-overlap(all_apis,h['end'],consume))
        for category, intervals in api_groups.items():
            row['return_'+category+'_ns'] = overlap(intervals,h['end'],consume)
    detail.append(row)
summary={'benchmark_valid':False,
         'definitions':{'input':'End of last submitted work on the stream before the CUDA event record synchronized for CPU inputs, through CPU entry. This is the configured part-A readiness boundary, not proof of earliest mathematically possible production.',
                        'return':'CPU return through the next MoE weighted-reduction kernel before the next CPU expert range. Includes copies, host scheduling/capture, preceding merge kernel and any remaining GPU hit work.',
                        'idle':'Intersection with whole-device idle; a zero-cost envelope, not a predicted speedup. Instrumented timings include profiler and preload overhead.',
                        'api':'CPU time inside runtime calls in the return window; overlaps device activity and must not be added to idle or other totals.'},
         'coverage':{'host_ranges':len(host),'input':sum('input_ready'in r for r in detail),'return':sum('reduction_start'in r for r in detail)},'by_tokens':{}}
fields = sorted({k for r in detail for k in r if k.endswith('_ns')})
summary['all']={f:describe([r[f] for r in detail if f in r]) for f in fields}
for t in sorted({r['tokens'] for r in detail}):
    summary['by_tokens'][str(t)]={f:describe([r[f] for r in detail if f in r and r['tokens']==t]) for f in fields}
(out/'transfer-detail.json').write_text(json.dumps(detail)+'\n')
(out/'transfer-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
