#!/usr/bin/env python3
"""Bounded, quiet upstream/fork stock-placement regression measurements."""
import argparse,json,os,pathlib,signal,subprocess,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--minutes',type=int,default=30);ap.add_argument('--out',default='stock-bench');ap.add_argument('--cases',nargs='+',choices=['moe-decode','moe-prefill','dense-gpu','dense-cpu']);a=ap.parse_args()
OUT=HERE/a.out;OUT.mkdir(exist_ok=False);end=time.monotonic()+a.minutes*60
UP=ROOT.parent/'llama-audit-upstream';HZ=os.sysconf('SC_CLK_TCK')
key=pathlib.Path('/opt/llm-gateway/data/api-key').read_text().strip()
def gw(method,path,body=None):
 req=urllib.request.Request('http://127.0.0.1:8790'+path,method=method,data=None if body is None else json.dumps(body).encode(),headers={'X-API-Key':key,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:return r.read()
def busy():
 v=list(map(int,pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:8]));return sum(v)-v[3]-v[4]
def own(pid):
 try:
  v=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return int(v[11])+int(v[12])+int(v[13])+int(v[14])
 except FileNotFoundError:return 0
def event(**v):
 v['utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
def quiet():
 while time.monotonic()<end:
  b=busy();time.sleep(2);foreign=(busy()-b)/HZ/2;load=os.getloadavg()[0]
  if load<2 and foreign<1:return load
  event(kind='wait_quiet',load=load,foreign_cores=foreign)
 raise TimeoutError('quiet-window budget exhausted')
q='/opt/llm-gateway/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
d='/opt/llm-gateway/models/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf'
cases={
 'moe-decode':['-m',q,'-ngl','99','-ncmoe','41','-p','0','-n','128','-d','128,12288'],
 'moe-prefill':['-m',q,'-ngl','99','-ncmoe','41','-p','12288','-n','0'],
 'dense-gpu':['-m',d,'-ngl','99','-p','0','-n','128'],
 'dense-cpu':['-m',d,'-ngl','0','-nopo','1','-p','0','-n','128']}
if a.cases: cases={name:cases[name] for name in a.cases}
shared=['-r','1','-o','json','-fa','on','-b','2048','-ub','512','-t','24','-lm','mmap']
plan=[dict(case=c,build=n,cmd=[str((UP if n=='upstream' else ROOT)/'build-cuda/bin/llama-bench')]+args+shared) for c,args in cases.items() for n in ('upstream','fork','fork','upstream','upstream','fork')]
(OUT/'plan.json').write_text(json.dumps({'plan':plan,'minutes':a.minutes,'note':'llama-bench bypasses the automatic expert cache; this is a stock-placement comparison. No semantic output comparison is supplied by synthetic llama-bench tokens.'},indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_MOE','GGML_CUDA_'))};env['LLAMA_MOE_STATE']='0'
def stop(*_):raise KeyboardInterrupt('SIGTERM')
signal.signal(signal.SIGTERM,stop)
held=False;child=None
try:
 for i,item in enumerate(plan):
  load=quiet()
  if not held:
   gw('POST','/api/v1/gpu/hold',{'reason':'codex: upstream fork regression','ttlS':900});held=True
   gw('POST','/api/v1/gpu/free',{})
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  gw('POST','/api/v1/gpu/hold',{'reason':'codex: upstream fork regression','ttlS':900})
  tag=f'{i:02d}-{item["case"]}-{item["build"]}'
  with (OUT/(tag+'.json')).open('w') as out,(OUT/(tag+'.txt')).open('w') as err:
   child=subprocess.Popen(item['cmd'],env=env,stdout=out,stderr=err,start_new_session=True)
   b0=busy();o0=own(child.pid);t0=time.monotonic();foreign_total=0;duration=0
   while child.poll() is None:
    time.sleep(1);now=time.monotonic();b1=busy();o1=own(child.pid)
    if o1:
     foreign_total+=max(0,b1-b0-(o1-o0))/HZ;duration+=now-t0
    b0,o0,t0=b1,o1,now
    if now>end:raise TimeoutError('benchmark budget exhausted')
   foreign=foreign_total/max(.001,duration)
   event(kind='run',index=i,tag=tag,**item,returncode=child.returncode,load_start=load,foreign_cores=foreign,accepted=child.returncode==0 and foreign<1)
   if child.returncode!=0:raise RuntimeError('benchmark failed: '+tag)
   if foreign>=1:raise RuntimeError('contaminated series: no aggregate comparison')
   child=None
 event(kind='complete')
except Exception as e:event(kind='incomplete',error=str(e))
finally:
 if child and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 if held:gw('DELETE','/api/v1/gpu/hold')
 event(kind='finished',gpu_released=True)
