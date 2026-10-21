#!/usr/bin/env python3
"""Capture one warm request with CUDA graph nodes, retaining owned-process cleanup."""
import argparse,ctypes,json,os,pathlib,signal,subprocess,sys,time
from native_reference import Reference,configs,bench,HERE
ap=argparse.ArgumentParser();ap.add_argument('--engine',choices=['strata','strata-reference'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--driver',action='store_true');ap.add_argument('--minutes',type=int,default=12);a=ap.parse_args()
out=HERE/a.out

def driver():
 run=Reference(out,time.time()+a.minutes*60)
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
 cfg=configs().get(a.engine)
 (out/'llama-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 nvtx=ctypes.CDLL('libnvToolsExt.so.1');nvtx.nvtxRangePushA.argtypes=[ctypes.c_char_p];nvtx.nvtxRangePushA.restype=ctypes.c_int;nvtx.nvtxRangePop.restype=ctypes.c_int
 try:
  (out/'owned-pids.json').write_text(json.dumps({'driver':os.getpid(),'owned':[]})+'\n')
  run.quiet();run.start(a.engine,cfg)
  tmp=out/'owned-pids.tmp';tmp.write_text(json.dumps({'driver':os.getpid(),'owned':run.owned()})+'\n');tmp.replace(out/'owned-pids.json')
  run.request('Say hello in five words.','warmup',16)
  prompt=bench.PROMPTS[0][1]
  run.quiet();run.event(kind='warm_request',**run.request(prompt,'novel',256))
  run.quiet();load=os.getloadavg()[0];b,o,t=bench.busy_jiffies(),bench.proc_jiffies(run.owned()),time.monotonic()
  nvtx.nvtxRangePushA(b'decode-audit')
  try:r=run.request(prompt,'profiled-repeat',256)
  finally:nvtx.nvtxRangePop()
  wall=time.monotonic()-t;foreign=max(0,bench.busy_jiffies()-b-bench.proc_jiffies(run.owned())+o)/bench.HZ/wall
  run.event(kind='profiled_request',load_start=load,foreign_cores=foreign,foreign_includes_profiler=True,quiet=load<2 and foreign<1,benchmark_valid=False,**r)
 finally:
  run.stop();run.event(kind='released',gpu=run.gpu_apps())

def outer():
 out.mkdir(exist_ok=False)
 run=Reference(out,time.time()+a.minutes*60);proc=None
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
 try:
  run.quiet();run.held=True;run.renew(force=True);bench.gw('POST','/api/v1/gpu/free',{})
  assert not run.gpu_apps(),'GPU has another owner'
  for sub in ['tmp','config','cache']:(out/sub).mkdir()
  env=dict(os.environ,TMPDIR=str(out/'tmp'),XDG_CONFIG_HOME=str(out/'config'),XDG_CACHE_HOME=str(out/'cache'))
  cmd=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=nvtx','--nvtx-capture=decode-audit','--capture-range-end=stop','--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0','--kill=none','--duration='+str(a.minutes*60),'--export=none','-o',str(out/'trace'),sys.executable,str(pathlib.Path(__file__).resolve()),'--engine',a.engine,'--out',a.out,'--minutes',str(a.minutes),'--driver']
  (out/'profile-command.json').write_text(json.dumps({'cmd':cmd,'benchmark_valid':False,'note':'Capture includes the short reused-prefix prompt processing and the generated response; do not treat profiler timings as benchmark rates.'},indent=2)+'\n')
  with (out/'nsys.txt').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+a.minutes*60+90
   while proc.poll() is None:
    run.renew()
    if time.monotonic()>deadline:raise TimeoutError('profile driver deadline')
    time.sleep(2)
  run.event(kind='profile_exit',returncode=proc.returncode)
 finally:
  if proc and proc.poll() is None:
   # Interrupt the Python driver first so its Reference lifecycle stops both engines.
   ids=out/'owned-pids.json'
   if ids.exists():
    p=json.loads(ids.read_text())['driver']
    try:
     cmdline=pathlib.Path(f'/proc/{p}/cmdline').read_bytes()
     if b'profile_engine.py' in cmdline and a.out.encode() in cmdline:os.kill(p,signal.SIGTERM)
    except FileNotFoundError:pass
   try:proc.wait(timeout=75)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGINT);proc.wait(timeout=15)
  run.stop();run.event(kind='outer_released',gpu=run.gpu_apps())

if a.driver:driver()
else:outer()
