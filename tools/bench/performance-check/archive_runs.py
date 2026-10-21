#!/usr/bin/env python3
"""Append audit runs without turning correctness-only timings into benchmark claims."""
import csv,datetime,hashlib,json,pathlib,shlex
ROOT=pathlib.Path(__file__).resolve().parents[3]; HERE=pathlib.Path(__file__).resolve().parent
path=ROOT/'tools/bench/run-history.csv'
with path.open(newline='') as f:
 reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
seen={r['campaign'] for r in old};rows=[]
hw='Threadripper3960X-24c48t_128GB_RTX2080Ti11GiB_sm75'
def add(tag,cmd,env,commit,ok,n=0,hit='',text='',note=''):
 campaign='performance-check-20261021/'+tag
 if campaign in seen:return
 r=dict.fromkeys(fields,'');r.update(ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit,model=cmd[cmd.index('-m')+1] if '-m' in cmd else '',test='audit-correctness',hw=hw,ok=str(int(ok)),build='Release-CUDA12.0-sm75',campaign=campaign,n_gen=str(n),origin='fork',env=json.dumps(env,sort_keys=True),cmd=shlex.join(cmd),args=shlex.join(cmd[1:]),hit=str(hit),md5=hashlib.md5(text.encode()).hexdigest() if text else '',note=note+'; NOT A PERFORMANCE BENCHMARK: tps/pp intentionally blank')
 rows.append(r)
for name,commit in [('fixture-before-off-fix','468a22f81'),('fixture-after-off-fix','aec2a5362'),('fixture-final','44f0ee12e')]:
 for cfg in sorted((HERE/name).glob('*-config.json')):
  c=json.loads(cfg.read_text());tag=cfg.name.removesuffix('-config.json');r=json.loads((cfg.parent/(tag+'.json')).read_text());mc=r['timings']['tuning']['moe_cache']
  add(name+'/'+tag,c['cmd'],c['env'],commit,not(name=='fixture-before-off-fix' and tag=='autotune-off'),r['tokens_predicted'],mc.get('hit_rate_request',''),r['content'],'tiny deterministic MoE fixture; source/model setup in make_fixture.py; raw '+str(cfg.relative_to(ROOT)))
for name,commit in [('mtp-integration','839353168'),('mtp-external-fix','08c52b62f'),('mtp-post-placement','d19629860')]:
 for r in json.loads((HERE/name/'summary.json').read_text()):
  p=HERE/name/(r['family']+'-'+r['mode']+'.json');response=json.loads(p.read_text()) if p.exists() else {};t=r.get('timings',{});mc=t.get('tuning',{}).get('moe_cache',{})
  add(name+'/'+r['family']+'-'+r['mode'],r['cmd'],r['env'],commit,r['ok'],r.get('tokens',0),mc.get('hit_rate_request',''),response.get('content',''),'functional completion only, ok does not assert text parity; draft '+str(t.get('draft_n_accepted',''))+'/'+str(t.get('draft_n',''))+'; source snapshot, embedded --version can be older')
with path.open('a',newline='') as f:
 writer=csv.DictWriter(f,fields,lineterminator="\n");writer.writerows(rows)
print('appended',len(rows),'functional runs; no throughput claims')
