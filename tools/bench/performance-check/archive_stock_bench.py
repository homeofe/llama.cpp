#!/usr/bin/env python3
"""Archive complete stock-control series and retain failed attempts without rates."""
import csv,json,pathlib,shlex,statistics
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
identities=json.loads((HERE/'build-identities.json').read_text())
path=ROOT/'tools/bench/run-history.csv'
with path.open(newline='') as f:
 reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
seen={r['campaign'] for r in old};rows=[];summary={}
for directory in sorted(HERE.glob('stock-bench-v[0-9]*')):
 if not directory.is_dir():continue
 events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
 assert events[-1]['kind']=='finished', 'do not archive a running series'
 runs={e['index']:e for e in events if e['kind']=='run'}
 contaminated=any(not e['accepted'] for e in runs.values())
 plan=json.loads((directory/'plan.json').read_text())['plan']
 complete_cases=set()
 for case in {e['case'] for e in plan}:
  group=[e for e in runs.values() if e['case']==case]
  if not contaminated and len(group)==6 and [e['build'] for e in group]==['upstream','fork','fork','upstream','upstream','fork'] and all(e['accepted'] and e['load_start']<2 and e['foreign_cores']<1 for e in group):complete_cases.add(case)
 for index,item in enumerate(plan):
  tag=f'{index:02d}-{item["case"]}-{item["build"]}';raw=directory/(tag+'.json')
  if not raw.exists():continue
  e=runs.get(index,{})
  try:data=json.loads(raw.read_text())
  except json.JSONDecodeError:data=[]
  valid=item['case'] in complete_cases and bool(data)
  for j,value in enumerate(data or [{}]):
   campaign='performance-check-20261021/'+directory.name+'/'+tag+'/'+str(j)
   if campaign in seen:continue
   cmd=item['cmd'];r=dict.fromkeys(fields,'')
   r.update(ts=value.get('test_time',e.get('utc',events[-1]['utc'])),commit=identities[item['build']]['source_commit'],bno=str(value.get('build_number','')),model=cmd[cmd.index('-m')+1],args=shlex.join(cmd[1:]),test='audit-stock-'+item['case'],hw='Threadripper3960X-24c48t_128GB_RTX2080Ti11GiB_sm75',ok=str(int(valid)),build='Release-CUDA12.0-sm75',campaign=campaign,n_gen=str(value.get('n_gen','')),origin=item['build'],env=json.dumps({'LLAMA_MOE_STATE':'0','LLAMA_*_inherited':'removed','GGML_MOE*_inherited':'removed','GGML_CUDA_*_inherited':'removed'},sort_keys=True),cmd=shlex.join(cmd),note=f'raw={raw.relative_to(ROOT)}; stock placement, no automatic cache; depth={value.get("n_depth","")}; load_start={e.get("load_start","")}; foreign_cores={e.get("foreign_cores","")}; '+('complete clean ABBAAB case; synthetic tokens, no semantic output comparison' if valid else 'INVALID for comparison: contaminated, incomplete or interrupted series; rates deliberately blank'))
   if valid:
    r['pp' if value['n_prompt'] else 'tps']=str(value['avg_ts'])
   rows.append(r)
 for case in sorted(complete_cases):
  values={}
  for e in runs.values():
   if e['case']!=case:continue
   for v in json.loads((directory/(e['tag']+'.json')).read_text()):
    values.setdefault((v['n_prompt'],v['n_gen'],v['n_depth']),{}).setdefault(e['build'],[]).append(v['avg_ts'])
  for (p,n,d),builds in values.items():
   assert all(len(v)==3 for v in builds.values()) and len(builds)==2
   stats={b:dict(n=len(v),median=statistics.median(v),min=min(v),max=max(v),samples=v) for b,v in builds.items()}
   summary[directory.name+'/'+case+f'/p{p}-n{n}-d{d}']=dict(**stats,fork_over_upstream=stats['fork']['median']/stats['upstream']['median'])
with path.open('a',newline='') as f:csv.DictWriter(f,fields,lineterminator="\n").writerows(rows)
(HERE/'stock-bench-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(appended=len(rows),summary=summary),indent=2))
