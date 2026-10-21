#!/usr/bin/env python3
"""Summarize a complete cache microbatch comparison and append its auditable rows."""
import argparse,csv,hashlib,json,pathlib,re,shlex,statistics
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
ap=argparse.ArgumentParser();ap.add_argument('directory',nargs='+',type=pathlib.Path);ap.add_argument('--out',type=pathlib.Path);a=ap.parse_args()
es=[];rows=[]
for d in a.directory:
 events=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()];assert events[-1]['kind']=='finished'
 es.extend(events)
 for r in json.loads((d/'rows.json').read_text()):r['_directory']=str(d);rows.append(r)
complete=sum(e['kind']=='block_complete' for e in es)==6 and not any(e['kind']=='incomplete' for e in es)
valid=complete and len(rows)==12 and all(r['accepted'] for r in rows)
if valid:
 assert [e['ub'] for e in es if e['kind']=='block_complete']==[512,1024,1024,512,512,1024]
 assert len({(r['block'],r['case']) for r in rows})==12
 for r in rows:
  t=r['timings']['tuning'];assert (t['n_threads'],t['n_threads_batch'],t['n_batch'],t['n_ubatch'])==(24,24,2048,r['ub'])
summary={'valid_comparison':valid,'protocol':'ABBAAB, n=3/configuration/workload, identical synthetic prompts, page cache prewarmed outside timing, fixed threads, autotuning off, fresh model/cache per block','cases':{},'output_pairs':[]}
if valid:
 for case in ['short','long']:
  entry={}
  for ub in [512,1024]:
   group=[r for r in rows if r['case']==case and r['ub']==ub];assert len(group)==3
   metrics={}
   for key in ['predicted_per_second','prompt_per_second']:
    v=[r['timings'][key] for r in group];metrics[key]={'n':3,'median':statistics.median(v),'min':min(v),'max':max(v),'samples':v}
   metrics['cache_slots']=[(r['timings']['tuning']['moe_cache']['slots_min'],r['timings']['tuning']['moe_cache']['slots_max']) for r in group]
   metrics['hit_rates']=[r['timings']['tuning']['moe_cache']['hit_rate_request'] for r in group]
   entry[str(ub)]=metrics
  for key in ['predicted_per_second','prompt_per_second']:entry[key+'_gain_pct']=100*(entry['1024'][key]['median']/entry['512'][key]['median']-1)
  summary['cases'][case]=entry
  for a_block,b_block in [(0,1),(3,2),(4,5)]:
   x=next(r for r in rows if r['block']==a_block and r['case']==case);y=next(r for r in rows if r['block']==b_block and r['case']==case)
   summary['output_pairs'].append({'case':case,'blocks':[a_block,b_block],'same_output':x['sha256']==y['sha256']})
(a.out or (a.directory[0]/'summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
p=ROOT/'tools/bench/run-history.csv'
with p.open(newline='') as f:
 reader=csv.DictReader(f);fields=reader.fieldnames;seen={r['campaign'] for r in reader}
identity=json.loads((HERE/'build-identities.json').read_text())['fork'];new=[]
for r in rows:
 d=pathlib.Path(r['_directory'])
 tag=f"{r['block']:02d}-ub{r['ub']}-{r['case']}";campaign='performance-check-20261021/'+d.name+'/'+tag
 if campaign in seen:continue
 response=json.loads((d/(tag+'.json')).read_text());cmd=r['cmd'];mc=r['timings']['tuning']['moe_cache']
 row=dict.fromkeys(fields,'');row.update(ts=next(e['utc'] for e in es if e['kind']=='request' and e['block']==r['block'] and e['case']==r['case']),commit=identity['source_commit'],bno=re.search(r'build (\d+)',identity['binary_version']).group(1),model=cmd[cmd.index('-m')+1],args=shlex.join(cmd[1:]),test='audit-cache-ub-'+r['case'],hw='Threadripper3960X-24c48t_128GB_RTX2080Ti11GiB_sm75',ok=str(int(valid)),build='Release-CUDA12.0-sm75',campaign=campaign,n_gen=str(r['tokens']),origin='fork',env=json.dumps(r['env'],sort_keys=True),cmd=shlex.join(cmd),md5=hashlib.md5(response['content'].encode()).hexdigest(),hit=str(mc['hit_rate_request']),note=f"raw={d.name}/{tag}.json; source and binary hashes in build-identities.json; ub={r['ub']}; slots={mc['slots_min']}..{mc['slots_max']}; load_start={r['load_start']}; foreign_cores={r['foreign_cores']}; "+('complete clean ABBAAB case; synthetic repeated long prompt' if valid else 'INCOMPLETE/INVALID series; rates blank'))
 if valid:row.update(tps=str(r['timings']['predicted_per_second']),pp=str(r['timings']['prompt_per_second']))
 new.append(row)
with p.open('a',newline='') as f:csv.DictWriter(f,fields,lineterminator='\n').writerows(new)
print(json.dumps({'appended':len(new),'summary':summary},indent=2))
