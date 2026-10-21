#!/usr/bin/env python3
"""Summarize recorded GPU interval unions; never add overlapping streams."""
import argparse, collections, json, pathlib, re, sqlite3

ap = argparse.ArgumentParser()
ap.add_argument('database', type=pathlib.Path)
ap.add_argument('--out', required=True, type=pathlib.Path)
a = ap.parse_args()
c = sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro', uri=True)
c.row_factory = sqlite3.Row
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
strings = dict(c.execute('select id,value from StringIds'))
def rows(table):
    return list(c.execute('select * from '+table)) if table in tables else []

kernels = rows('CUPTI_ACTIVITY_KIND_KERNEL')
copies = rows('CUPTI_ACTIVITY_KIND_MEMCPY')
sets = rows('CUPTI_ACTIVITY_KIND_MEMSET')
assert kernels, 'No CUDA kernels recorded; do not interpret an empty trace as idle.'
ranges = [r for r in rows('NVTX_EVENTS') if (r['text'] or strings.get(r['textId'])) == 'decode-audit' and r['end'] is not None]
assert len(ranges) <= 1, 'Unexpected repeated capture ranges.'
activity = kernels+copies+sets
lo, hi = (ranges[0]['start'], ranges[0]['end']) if ranges else (min(r['start'] for r in activity), max(r['end'] for r in activity))
assert hi > lo
def intervals(items):
    return [(max(lo,r['start']),min(hi,r['end'])) for r in items if r['start']<hi and r['end']>lo]
def union(items):
    total=0; start=end=None
    for x,y in sorted(items):
        if start is None: start,end=x,y
        elif x>end: total+=end-start;start,end=x,y
        else: end=max(end,y)
    return total+(end-start if start is not None else 0)
def name(r): return strings[r['demangledName']]
wait_re = re.compile(r'wait_flag_ge(?:_or)?_kernel|doorbell_wait_kernel')
wait = [r for r in kernels if wait_re.search(name(r))]
work = [r for r in kernels if not wait_re.search(name(r))]
busy = union(intervals(activity)); productive=union(intervals(work+copies+sets))
top = collections.defaultdict(lambda: {'count':0,'summed_ms':0.0})
for r in kernels:
    duration=max(0,min(hi,r['end'])-max(lo,r['start']))
    if duration:
        t=top[name(r)];t['count']+=1;t['summed_ms']+=duration/1e6
api={}
for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
    group=collections.defaultdict(lambda: {'count':0,'summed_ms':0.0})
    for r in rows(table):
        if r['start']<hi and r['end']>lo:
            t=group[strings[r['nameId']]];t['count']+=1;t['summed_ms']+=(min(hi,r['end'])-max(lo,r['start']))/1e6
    api[table]=dict(sorted(group.items(),key=lambda p:-p[1]['summed_ms'])[:20])
result={
    'source':str(a.database), 'benchmark_valid':False,
    'window':'complete decode-audit NVTX range' if ranges else 'first-to-last recorded GPU activity; NVTX range incomplete',
    'note':'Capture includes reused-prefix processing. Kernel/API sums are descriptive and overlap; only interval unions partition device time. Non-wait kernels are not a proof of useful arithmetic. No draft/target attribution is inferred from shared quantization names.',
    'window_ms':(hi-lo)/1e6, 'gpu_activity_ms':busy/1e6,
    'non_wait_kernel_copy_memset_union_ms':productive/1e6,
    'wait_only_ms':(busy-productive)/1e6, 'device_idle_ms':(hi-lo-busy)/1e6,
    'kernel_count':len(kernels), 'graph_kernel_count':sum(bool(r['graphNodeId']) for r in kernels),
    'wait_kernel_count':len(wait), 'copy_count':len(copies),
    'copy_bytes_by_kind':dict(collections.Counter({kind:sum(r['bytes'] for r in copies if r['copyKind']==kind) for kind in {r['copyKind'] for r in copies}})),
    'top_kernel_summed_ms':dict(sorted(top.items(),key=lambda p:-p[1]['summed_ms'])[:25]),
    'host_api_summed_ms':api,
}
assert abs(result['window_ms']-sum(result[k] for k in ['non_wait_kernel_copy_memset_union_ms','wait_only_ms','device_idle_ms']))<1e-6
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['top_kernel_summed_ms','host_api_summed_ms']},indent=2))
