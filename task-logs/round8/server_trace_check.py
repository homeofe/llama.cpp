#!/usr/bin/env python3
"""Exercise server verdict records with two small CPU-only MoE fixtures."""
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parents[2]
logs = root / 'task-logs/round8'
trace = logs / 'server-cpu-trace.jsonl'
trace.unlink(missing_ok=True)
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
command = [str(root/'build-cpu/bin/llama-server'),
           '-m', str(root/'task-logs/round7/models/qwen3moe-moe.gguf'),
           '-md', str(logs/'models/qwen3moe-moe.gguf'),
           '--host', '127.0.0.1', '--port', str(port), '--no-webui',
           '-ngl', '0', '-ngld', '0', '-t', '4', '-tb', '4', '-c', '128', '-b', '64', '-ub', '16',
           '--spec-type', 'draft-simple', '--spec-draft-n-max', '3', '--spec-draft-n-min', '1',
           '--spec-draft-p-min', '0', '--spec-draft-n-ubatch', '4']
env = {k:v for k,v in os.environ.items() if not k.startswith('GGML_MOE_HYBRID')}
env.update(GGML_MOE_HYBRID_TRACE=str(trace), GGML_BACKEND_PATH=str(root/'build-cpu/bin'))
with (logs/'server-cpu.log').open('w') as output:
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + 90
        while True:
            if child.poll() is not None:
                raise RuntimeError(f'CPU server exited: {child.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('CPU server startup timeout')
            time.sleep(0.2)
        responses = []
        for start in [1,40]:
            body = dict(prompt=list(range(start,start+20)), n_predict=24, temperature=0,
                        cache_prompt=False, ignore_eos=True, seed=17)
            request = urllib.request.Request(f'http://127.0.0.1:{port}/completion',
                data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=90) as response:
                responses.append(json.load(response))
        (logs/'server-cpu-responses.json').write_text(json.dumps(responses,indent=2)+'\n')
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

records = [json.loads(line) for line in trace.read_text().splitlines()]
target = next(r['context'] for r in records if 'layers' in r and not r['draft'])
spans = {}
begins = rejected = 0
for record in records:
    if record['context'] != target or 'event' not in record:
        continue
    seq = record['seq']
    if record['event'] == 'spec_begin':
        assert seq not in spans
        spans[seq] = record
        begins += 1
    else:
        begin = spans.pop(seq)
        assert begin['pos'] == record['pos'] and begin['n_tokens'] == record['n_tokens']
        assert 1 <= record['n_accepted'] <= record['n_tokens']
        rejected += record['n_tokens'] - record['n_accepted']
assert not spans and begins > 0 and rejected > 0
replay = subprocess.check_output([str(root/'build-cpu/bin/test-moe-cache'), '--replay-log', str(trace),
    '2','1','2','0','64','-',str(target)], text=True, env={**env,'GGML_MOE_REPLAY_ACCEPTED_ONLY':'1'})
(logs/'server-cpu-replay.log').write_text(replay)
metrics = dict(re.findall(r'([\w/]+)=([^\s]+)',next(line for line in replay.splitlines() if line.startswith('context='))))
assert metrics['unresolved_spans'] == '0' and int(metrics['rejected_routes']) > 0
print(f'CPU server: {begins} closed target spans, {rejected} rejected input positions; causal replay passed')
