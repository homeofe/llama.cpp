#!/usr/bin/env python3
"""Real Qwen/GLM MTP smoke checks. No rates from these loaded-machine runs are benchmarks."""
import argparse,hashlib,json,os,pathlib,signal,subprocess,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[3]; HERE=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--out',default='mtp-integration');ap.add_argument('--families',nargs='+',default=['qwen','glm']);ap.add_argument('--modes',nargs='+',default=['plain','mtp','auto']);a=ap.parse_args()
OUT=HERE/a.out;OUT.mkdir(exist_ok=False)
key=pathlib.Path('/opt/llm-gateway/data/api-key').read_text().strip()
def gw(method,path,body=None):
 req=urllib.request.Request('http://127.0.0.1:8790'+path,method=method,data=None if body is None else json.dumps(body).encode(),headers={'X-API-Key':key,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:return r.read()
def api(path,body=None):
 req=urllib.request.Request('http://127.0.0.1:18123'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
models={
 'qwen':('/opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf','/opt/llm-gateway/models/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'),
 'glm':('/opt/llm-gateway/models/GLM-5.3-Flash-GGUF/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf','/opt/llm-gateway/data/moe-quant/audit/GLM-5.3-Flash-MTP-Q4_K.gguf')}
results=[]
def run(family,mode):
 model,draft=models[family]
 name=family+'-'+mode
 cmd=[str(ROOT/'build-cuda/bin/llama-server'),'-m',model,'-c','2048','-np','1','-b','256','-ub','64','-t','24','-tb','24','-fa','1','--load-mode','mmap','--no-warmup','--host','127.0.0.1','--port','18123']
 if mode!='plain':cmd+=['-md',draft,'--spec-type','draft-mtp','-ngld','0']
 if mode=='mtp':cmd+=['--spec-draft-n-max','3']
 env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_MOE','LLAMA_MOE','LLAMA_ARG','LLAMA_SPEC','LLAMA_AUTOTUNE'))}
 env.update(LLAMA_MOE_STATE='0')
 if mode=='mtp':env['LLAMA_SPEC_DEPTH']='3'
 row={'family':family,'mode':mode,'cmd':cmd,'env':{k:v for k,v in env.items() if k.startswith('LLAMA_')},'benchmark_valid':False}
 (OUT/(name+'-config.json')).write_text(json.dumps(row,indent=2)+'\n')
 with (OUT/(name+'.txt')).open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   end=time.time()+240
   while time.time()<end:
    if p.poll() is not None:raise RuntimeError('server exited '+str(p.returncode))
    try:
     if api('/health').get('status')=='ok':break
    except Exception:pass
    time.sleep(1)
   else:raise TimeoutError('model load')
   r=api('/completion',dict(prompt='Complete this Python function. Return only code.\ndef fibonacci(n):\n',n_predict=64,ignore_eos=True,temperature=0,seed=1,cache_prompt=False))
   (OUT/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
   row.update(ok=True,sha256=hashlib.sha256(r['content'].encode()).hexdigest(),tokens=r['tokens_predicted'],timings=r['timings'])
  except Exception as e:row.update(ok=False,error=str(e))
  finally:
   p.terminate()
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   row['draft_not_loaded_message']='not loading it' in (OUT/(name+'.txt')).read_text()
 return row
def stop(*_):raise KeyboardInterrupt('SIGTERM')
signal.signal(signal.SIGTERM,stop)
gw('POST','/api/v1/gpu/hold',{'reason':'codex: Qwen and GLM MTP correctness','ttlS':1800})
try:
 gw('POST','/api/v1/gpu/free',{})
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 for family in a.families:
  for mode in a.modes:
   row=run(family,mode);results.append(row)
   (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
   print(family,mode,row.get('ok'),row.get('error',''),flush=True)
finally:gw('DELETE','/api/v1/gpu/hold')
