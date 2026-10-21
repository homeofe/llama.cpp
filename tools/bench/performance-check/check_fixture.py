#!/usr/bin/env python3
"""GPU integration checks on a tiny local fixture. Timings are not benchmarks."""
import hashlib, json, os, pathlib, signal, subprocess, sys, time, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
PORT = 18122
LABEL = sys.argv[1]
OUT = HERE/LABEL
OUT.mkdir(exist_ok=True)
STATE = OUT/'moe-state.ini'
KEY = pathlib.Path('/opt/llm-gateway/data/api-key').read_text().strip()

def gw(method, path, body=None):
    req = urllib.request.Request('http://127.0.0.1:8790'+path, method=method, data=None if body is None else json.dumps(body).encode(), headers={'X-API-Key':KEY, 'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r:
        return r.read()

def api(path, body=None):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}'+path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r: return json.load(r)

def run(name, args, state=True):
    cmd = [str(ROOT/'build-cuda/bin/llama-server'),'-m',str(ROOT/'build-cuda/audit-moe.gguf'),'-np','1','-c','1024','-b','128','-ub','64','-t','2','-tb','2','-fa','1','-fit','off','--no-warmup','--host','127.0.0.1','--port',str(PORT)] + args
    env = dict(os.environ, LLAMA_MOE_STATE=str(STATE) if state else '0')
    for k in list(env):
        if k.startswith(('LLAMA_MOE_CACHE_', 'GGML_MOE_', 'LLAMA_ARG_')): env.pop(k)
    (OUT/(name+'-config.json')).write_text(json.dumps({'cmd':cmd,'env':{'LLAMA_MOE_STATE':env['LLAMA_MOE_STATE']},'benchmark':False},indent=2)+'\n')
    with (OUT/(name+'.txt')).open('w') as log:
        p = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            end=time.time()+90
            while True:
                assert p.poll() is None, f'{name}: server exited {p.returncode}'
                try:
                    if api('/health').get('status') == 'ok': break
                except Exception: pass
                assert time.time()<end, f'{name}: load timed out'
                time.sleep(.2)
            r = api('/completion',dict(prompt=[1,22,31,42],n_predict=80,ignore_eos=True,temperature=0,seed=1,cache_prompt=False))
            (OUT/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
            return r
        finally:
            p.terminate()
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: p.kill();p.wait()

gw('POST','/api/v1/gpu/hold',{'reason':'codex: fork performance audit correctness','ttlS':900})
try:
    gw('POST','/api/v1/gpu/free',{})
    occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    assert not occupied, 'GPU has another owner'
    if STATE.exists(): STATE.unlink()
    base=['-ngl','99','--cpu-moe','--no-repack','--moe','cache=4']
    seeded=run('seed',base+['-at','on'])
    info=seeded['timings']['tuning']['moe_cache']
    assert info['active'] and info['slots_min']==4 and info['slots_max']==4, info
    assert STATE.exists() and 'hot.0' in STATE.read_text(), 'hot state was not saved after 80 tokens'
    text=STATE.read_text()
    text += 'tuned.l1 = GATE=123\n'
    STATE.write_text(text)
    before=(STATE.read_bytes(),STATE.stat().st_mtime_ns)
    off=run('autotune-off',base+['-at','off'])
    info=off['timings']['tuning']['moe_cache']
    failures=[]
    if 'saved settings loaded from the state file' in (OUT/'autotune-off.txt').read_text(): failures.append('autotune-off loaded saved tuning settings')
    if before!=(STATE.read_bytes(),STATE.stat().st_mtime_ns): failures.append('autotune-off wrote the state file')
    for name,args in [('fits',[]),('cache-zero',base+['--moe','cache=0','-at','on'])]:
        r=run(name,args)
        assert not r['timings']['tuning']['moe_cache']['active'], name
    auto=run('auto-slots',base+['--moe-expert-cache','-1','-at','off'],False)
    ai=auto['timings']['tuning']['moe_cache']
    assert ai['active'] and ai['slots_min']>0 and ai['slots_max']<=8, ai
    (OUT/'summary.json').write_text(json.dumps({'failures':failures,'fixed_slots':4,'auto_slots':ai['slots_max'],'fit_cache_off':True,'cache_zero_off':True,'benchmark_valid':False},indent=2)+'\n')
    print(json.dumps({'failures':failures,'benchmark_valid':False}))
    sys.exit(bool(failures))
finally:
    gw('DELETE','/api/v1/gpu/hold')
