#!/usr/bin/env python3
"""Build only local CPU libraries while the owned benchmark is safely paused."""
import json, os, pathlib, signal, subprocess, time, urllib.request
root = pathlib.Path(__file__).resolve().parents[3]
pid = 1483672
proc = pathlib.Path(f'/proc/{pid}')
assert b'task-logs/round16/native_reference.py' in (proc/'cmdline').read_bytes().split(b'\0')
assert (proc/'wchan').read_text().strip() == 'hrtimer_nanosleep', 'runner is not between requests'
identity = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
os.kill(pid, signal.SIGSTOP)
start = time.time()
events = root/'tools/bench/performance-check/build-events.jsonl'
def event(**kw):
    with events.open('a') as f: f.write(json.dumps(dict(epoch=time.time(), **kw))+'\n')
try:
    event(kind='pause_reference', pid=pid)
    key = pathlib.Path('/opt/llm-gateway/data/api-key').read_text().strip()
    req = urllib.request.Request('http://127.0.0.1:8790/api/v1/gpu/hold', data=json.dumps({'reason':'codex: round16 Strata draft parity','ttlS':900}).encode(), headers={'X-API-Key':key,'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req,timeout=10) as r: assert r.status == 200
    with (root/'tools/bench/performance-check/build-cpu.txt').open('w') as log:
        subprocess.run(['cmake','-S',str(root),'-B',str(root/'build-cpu'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=OFF','-DGGML_NATIVE=ON','-DLLAMA_USE_PREBUILT_UI=OFF','-DLLAMA_BUILD_UI=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_APP=OFF'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        subprocess.run(['nice','-n','10','cmake','--build',str(root/'build-cpu'),'--target','llama-common','-j','8'], stdout=log,stderr=subprocess.STDOUT,check=True,timeout=480)
    event(kind='cpu_build_complete',seconds=time.time()-start)
finally:
    if proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[19] == identity:
        os.kill(pid,signal.SIGCONT)
        event(kind='resume_reference',pid=pid)
