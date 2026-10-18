#!/usr/bin/env python3
# Qwen3.8-Flash-Next in llama.cpp after the Bolt work (2026-10-17): the stage A harness of the Strata-parity plan
# (patches/llama.cpp/strata-parity/stage_a.py, same prompts, same gating, same client-side speed), with the new fork
# (worktree llama-l4: l3-mmq, i.e. the Bolt stack, plus the IQ3 AVX2 kernels) against the stage A build and Strata as
# deployed, on the same clock.
#
# usage: stage_b.py OUT_DIR DEADLINE_HHMM_UTC [--only CONFIG ...] [--rounds N]
import argparse, calendar, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

KEY = Path('/opt/llm-gateway/data/api-key').read_text().strip()
GW = 'http://127.0.0.1:8790'
HZ = os.sysconf('SC_CLK_TCK')
MAX_WAIT_S = 1800
HELD_WAIT_S = 300   # longer waits for quiet give the GPU back to the gateway first


class Requeue(Exception):
    pass
OLD = '/opt/llm-gateway/data/build/llama-codex4/build-parity-cuda/bin/llama-server'
NEW = '/opt/llm-gateway/data/build/llama-l4/build-cuda/bin/llama-server'
L5 = '/opt/llm-gateway/data/build/llama-l5/build-cuda/bin/llama-server'
CACHE = {'GGML_MOE_HYBRID': '1', 'GGML_MOE_HYBRID_MIB': '2048',
         'GGML_MOE_HYBRID_PROFILE': '/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1-2048mib.profile'}
MODEL = '/opt/llm-gateway/models/Qwen3.8-Flash-Next/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
MTP = '/opt/llm-gateway/models/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
PORT = 8799
ARGS = ['-m', MODEL, '--host', '127.0.0.1', '--port', str(PORT), '--no-webui', '-fa', 'on', '-c', '8192', '-np', '1',
        '-ub', '512', '-b', '2048', '-t', '24', '-tb', '24', '-ngl', '99', '-ncmoe', '48']
UB256 = [('256' if a == '512' else a) for a in ARGS]
ADAPT = {'GGML_MOE_HYBRID': '1', 'GGML_MOE_HYBRID_ADAPTIVE': '1', 'GGML_MOE_HYBRID_ADMIT': '32'}
PROF2G = '/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1-2048mib.profile'
PROF3G = '/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1-3072mib.profile'
LING = ['-m', '/opt/llm-gateway/models/Ling-3.0-flash-GGUF/Ling-3.0-flash-IQ4_XS/Ling-3.0-flash-IQ4_XS-00001-of-00002.gguf',
        '--host', '127.0.0.1', '--port', str(PORT), '--no-webui', '--jinja', '-fa', 'on', '-c', '8192', '-np', '1',
        '-ub', '512', '-b', '2048', '-t', '24', '-tb', '24', '-ngl', '99', '-cmoe']
MTPA = ['--spec-type', 'draft-mtp', '-md', MTP, '-cmoed']
# the Bolt switches: shared expert beside the CPU experts, overlap of the GPU split before a CPU split, prompts under
# 256 tokens on the CPU; -lm none keeps the experts in pinned memory
BOLT = {'GGML_SCHED_OVERLAP': '1', 'LLAMA_MOE_SHEXP_EARLY': '1', 'GGML_OP_OFFLOAD_MIN_BATCH': '256',
        'OMP_PLACES': 'cores', 'OMP_PROC_BIND': 'close'}
CONFIGS = {
    'old-mtp3-pin':     dict(cmd=[OLD] + ARGS + MTPA + ['--spec-draft-n-max', '3'], env={'OMP_PLACES': 'cores', 'OMP_PROC_BIND': 'close'}),
    'new-mtp3-pin':     dict(cmd=[NEW] + ARGS + MTPA + ['--spec-draft-n-max', '3'], env={'OMP_PLACES': 'cores', 'OMP_PROC_BIND': 'close'}),
    'new-mtp3-bolt':    dict(cmd=[NEW] + ARGS + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'], env=BOLT),
    'new-mtp2-bolt':    dict(cmd=[NEW] + ARGS + MTPA + ['--spec-draft-n-max', '2', '-lm', 'none'], env=BOLT),
    # the BF16 hyper-connection matrices and the router retyped for Turing (retype.py: hc down Q8_0, hc up and router F16)
    'hcfix-mtp3-bolt':  dict(cmd=[NEW] + [a.replace('/IQ3_S/', '/IQ3_S-hcfix/') for a in ARGS] + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'], env=BOLT),
    # round 5 (worktree llama-l5): the same binary without and with the frozen GPU-hit / CPU-miss cache (2 GB profile
    # from the round 4 trace; the MTP head leaves about 3 GB of VRAM)
    'l5-mtp3-bolt':     dict(cmd=[L5] + ARGS + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'], env=BOLT),
    'l5-cache-mtp3':    dict(cmd=[L5] + ARGS + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'], env={**BOLT, **CACHE}),
    # round 6: adaptive LFU admission, warm start from the 2 GB profile (which also keeps the draft model's context off)
    'l5-adapt-mtp3':    dict(cmd=[L5] + ARGS + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'],
                             env={**BOLT, **CACHE, 'GGML_MOE_HYBRID_ADAPTIVE': '1', 'GGML_MOE_HYBRID_ADMIT': '32'}),
    # -ub 256 halves both compute buffers (main and MTP draft), which leaves 3 GB for the adaptive cache
    'l5-adapt3g-ub256': dict(cmd=[L5] + [('256' if a == '512' else a) for a in ARGS] + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'],
                             env={**BOLT, 'GGML_MOE_HYBRID': '1', 'GGML_MOE_HYBRID_MIB': '3072', 'GGML_MOE_HYBRID_ADAPTIVE': '1', 'GGML_MOE_HYBRID_ADMIT': '32',
                                  'GGML_MOE_HYBRID_PROFILE': '/opt/llm-gateway/data/build/llama-codex2/task-logs/round4-hitmiss/qwen-routes-r1-3072mib.profile'}),
    'l5-mtp3-ub256':    dict(cmd=[L5] + [('256' if a == '512' else a) for a in ARGS] + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'], env=BOLT),
    # round 7: prompt warming (up to 1024 experts after the prompt) and faster decay; then a separate draft cache
    'q-warm':           dict(cmd=[L5] + UB256 + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'],
                             env={**BOLT, **ADAPT, 'GGML_MOE_HYBRID_MIB': '3072', 'GGML_MOE_HYBRID_PROFILE': PROF3G,
                                  'GGML_MOE_HYBRID_PREFILL': '1', 'GGML_MOE_HYBRID_DECAY': '128'}),
    'q-warm-draft':     dict(cmd=[L5] + UB256 + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'],
                             env={**BOLT, **ADAPT, 'GGML_MOE_HYBRID_MIB': '2560', 'GGML_MOE_HYBRID_PROFILE': PROF2G,
                                  'GGML_MOE_HYBRID_PREFILL': '1', 'GGML_MOE_HYBRID_DECAY': '128', 'GGML_MOE_HYBRID_DRAFT_MIB': '512'}),
    # Ling-3.0-flash (bailingmoe3, MTP head in the same file): all experts in RAM, then with a 5 GB adaptive cache
    'ling-mtp3':        dict(cmd=[L5] + LING + ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '-lm', 'none'], env=BOLT),
    'ling-cache-mtp3':  dict(cmd=[L5] + LING + ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '-lm', 'none'],
                             env={**BOLT, **ADAPT, 'GGML_MOE_HYBRID_MIB': '5120', 'GGML_MOE_HYBRID_PREFILL': '1', 'GGML_MOE_HYBRID_DECAY': '128'}),
    # route traces of the real benchmark for round 8 (same settings as q-warm-draft and ling-cache-mtp3)
    'q-trace':          dict(cmd=[L5] + UB256 + MTPA + ['--spec-draft-n-max', '3', '-lm', 'none'],
                             env={**BOLT, **ADAPT, 'GGML_MOE_HYBRID_MIB': '2560', 'GGML_MOE_HYBRID_PROFILE': PROF2G,
                                  'GGML_MOE_HYBRID_PREFILL': '1', 'GGML_MOE_HYBRID_DECAY': '128', 'GGML_MOE_HYBRID_DRAFT_MIB': '512',
                                  'GGML_MOE_HYBRID_TRACE': '/opt/llm-gateway/data/moe-quant/traces/qwen38-bench8.jsonl'}),
    'ling-trace':       dict(cmd=[L5] + LING + ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '-lm', 'none'],
                             env={**BOLT, **ADAPT, 'GGML_MOE_HYBRID_MIB': '5120', 'GGML_MOE_HYBRID_PREFILL': '1', 'GGML_MOE_HYBRID_DECAY': '128',
                                  'GGML_MOE_HYBRID_TRACE': '/opt/llm-gateway/data/moe-quant/traces/ling-bench8.jsonl'}),
    'strata':           dict(gateway_model='iq3_s'),
}
# Round 8 candidates. The policy and buffer changes are opt-in; existing controls stay intact.
for name, control, extra, env in [
    ('q-r8-byte', 'q-warm-draft', [], {'GGML_MOE_HYBRID_BYTE_VALUE': '1'}),
    ('ling-r8-byte', 'ling-cache-mtp3', [], {'GGML_MOE_HYBRID_BYTE_VALUE': '1'}),
    ('q-r8-policy', 'q-warm-draft', [], {'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
    ('ling-r8-policy', 'ling-cache-mtp3', [], {'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
    ('q-r8-ub32', 'q-warm-draft', ['--spec-draft-n-ubatch', '32'], {}),
    ('ling-r8-ub32', 'ling-cache-mtp3', ['--spec-draft-n-ubatch', '32'], {}),
    ('q-r8-slots', 'q-warm-draft', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '2944'}),
    ('ling-r8-slots', 'ling-cache-mtp3', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '5376'}),
    ('q-r8-combined', 'q-warm-draft', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '2944', 'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
    ('ling-r8-combined', 'ling-cache-mtp3', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '5376', 'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
]:
    CONFIGS[name] = dict(cmd=CONFIGS[control]['cmd'] + extra, env={**CONFIGS[control]['env'], **env})

ENGLISH = [
    'Explain the difference between a process and a thread, and when you would choose one over the other in a web server.',
    'Summarize the trade-offs of putting a message queue between two services: what it buys you and what it costs.',
    'Write the "Usage" section of a README for a command-line tool that syncs a local folder to an S3 bucket.',
    'Explain how HTTP caching works with Cache-Control, ETag and If-None-Match, with one short example exchange.',
]


def code_prompts():
    src = Path('/opt/llm-gateway/apps/bench/src/hard.ts').read_text()
    body = src[src.index('const CODE'):src.index('];', src.index('const CODE'))]
    out = []
    for m in re.finditer(r'\{ id: "([a-z]+)", prompt: ("(?:[^"\\]|\\.)*")', body):
        out.append((m[1], json.loads(m[2]) + '\nReturn only the code, in one ```python block, without tests or examples.'))
    return out[:4]


PROMPTS = [('code-' + i, p) for i, p in code_prompts()] + [(f'en-{k}', p) for k, p in enumerate(ENGLISH, 1)]


def busy_jiffies():
    v = list(map(int, open('/proc/stat').readline().split()[1:]))
    return sum(v[:8]) - v[3] - v[4]


def proc_jiffies(pids):
    t = 0
    for p in pids:
        try:
            f = open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()
            t += int(f[11]) + int(f[12])
        except (FileNotFoundError, ProcessLookupError, IndexError):
            pass
    return t


def engine_pids(name):
    # the engine under test: llama-server on our port, or every Strata process (read from /proc, nothing is signalled)
    pids = []
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            cmd = open(f'/proc/{d}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if (name == 'strata' and '/Strata-wt-041/' in cmd) or (name != 'strata' and 'llama-server' in cmd and str(PORT) in cmd):
            pids.append(int(d))
    return pids


def gw(method, path, payload=None, timeout=30):
    r = urllib.request.Request(GW + path, method=method, data=json.dumps(payload).encode() if payload is not None else None,
                               headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return f.read().decode()


class Run:
    def __init__(self, out, end, load_gate=False):
        self.out, self.end = out, end
        self.load_gate = load_gate
        self.gpu_pid = None
        self.previous_gpu_pids = set()
        self.log = out / 'runs.jsonl'
        self.done = set()
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                x = json.loads(line)
                if x.get('kind') == 'request' and x.get('accepted'):
                    self.done.add(x['label'])
        self.child = None
        self.held = False
        self.last_hold = 0

    def event(self, **kw):
        kw.update(epoch=round(time.time(), 1), utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
        with self.log.open('a') as f:
            f.write(json.dumps(kw) + '\n')
        print(json.dumps(kw), flush=True)

    def renew(self, force=False):
        if self.held and (force or time.monotonic() - self.last_hold > 300):
            gw('POST', '/api/v1/gpu/hold', {'reason': 'codex: round 8' if self.load_gate else 'claude: strata-parity stage A (llama.cpp)', 'ttlS': 900})
            self.last_hold = time.monotonic()

    def quiet(self):
        t0 = time.time()
        while True:
            if time.time() > self.end:
                raise TimeoutError('deadline')
            self.renew()
            if self.load_gate:
                cores = os.getloadavg()[0]
                if cores < 3:
                    if self.child is not None:
                        self.check_gpu()
                    return round(time.time() - t0), round(cores, 2)
                if self.held:
                    raise Requeue('load gate')
            else:
                b0 = busy_jiffies(); time.sleep(5)
                cores = (busy_jiffies() - b0) / HZ / 5
                if cores < 1.0:
                    return round(time.time() - t0), round(cores, 2)
            if self.held and time.time() - t0 > HELD_WAIT_S:
                raise Requeue()
            if time.time() - t0 > MAX_WAIT_S:
                raise TimeoutError(f'no quiet window within {MAX_WAIT_S} s')
            time.sleep(10)

    def gpu_apps(self):
        text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        return {int(row.split(',')[0]): int(row.split(',')[1]) for row in text.splitlines() if row.strip()}

    def check_gpu(self):
        current = self.gpu_apps()
        new_pids = set(current) - self.previous_gpu_pids
        if self.gpu_pid is None and len(new_pids) == 1:
            self.gpu_pid = next(iter(new_pids))
        foreign = {pid: mib for pid, mib in current.items() if pid != self.gpu_pid and mib > 300}
        if foreign:
            raise Requeue('foreign GPU process: ' + str(foreign))

    def start(self, name, cfg):
        if 'gateway_model' in cfg:
            return
        if self.load_gate:
            self.gpu_apps()
        gw('POST', '/api/v1/gpu/free', {})
        self.held = True
        self.renew(force=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        if any(int(l.split(',')[1]) > 300 for l in apps.splitlines() if l.strip()):
            raise RuntimeError('GPU in use: ' + apps.strip())
        self.previous_gpu_pids = set(self.gpu_apps()) if self.load_gate else set()
        log = open(self.out / f'{name}.server.log', 'a')
        env = dict(os.environ)
        if self.load_gate:
            for key in list(env):
                if key.startswith('GGML_MOE_HYBRID') or key == 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH':
                    del env[key]
        self.child = subprocess.Popen(cfg['cmd'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      env={**env, **cfg.get('env', {})}, start_new_session=True)
        log.close()
        t0 = time.time()
        while True:
            if self.child.poll() is not None:
                raise RuntimeError(f'{name}: server exited {self.child.returncode}')
            if self.load_gate:
                self.check_gpu()
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as f:
                    if f.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError(f'{name}: not healthy after 900 s')
            self.renew()
            time.sleep(2)
        if self.load_gate:
            self.check_gpu()
            if self.gpu_pid is None:
                raise RuntimeError('cannot identify the server GPU PID')

    def stop(self):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL); self.child.wait()
        self.child = None
        self.gpu_pid = None
        if self.held:
            gw('DELETE', '/api/v1/gpu/hold')
            self.held = False

    def request(self, name, cfg, text, max_tokens):
        body = dict(messages=[{'role': 'user', 'content': text}], max_tokens=max_tokens, temperature=0, stream=True,
                    stream_options={'include_usage': True}, chat_template_kwargs={'enable_thinking': False})
        if 'gateway_model' in cfg:
            url, headers = GW + '/v1/chat/completions', {'Authorization': 'Bearer ' + KEY}
            body['model'] = cfg['gateway_model']
        else:
            url, headers = f'http://127.0.0.1:{PORT}/v1/chat/completions', {}
        r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **headers})
        t_first = t_last = None; pieces = 0; usage = None; timings = None; text_out = []
        t0 = time.time()
        last_gpu_check = time.monotonic()
        with urllib.request.urlopen(r, timeout=1800) as f:
            for raw in f:
                if self.load_gate and time.monotonic() - last_gpu_check > 5:
                    self.check_gpu()
                    self.renew()
                    last_gpu_check = time.monotonic()
                line = raw.decode(errors='replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                d = json.loads(data)
                if d.get('usage'):
                    usage = d['usage']
                if d.get('timings'):
                    timings = d['timings']
                for ch in d.get('choices') or []:
                    c = (ch.get('delta') or {}).get('content') or (ch.get('delta') or {}).get('reasoning_content')
                    if c:
                        now = time.time()
                        t_first = t_first or now
                        t_last = now
                        pieces += 1
                        text_out.append(c)
        n = (usage or {}).get('completion_tokens') or pieces
        rate = (n - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and n > 1 else None
        return dict(tokens=n, pieces=pieces, decode_tok_s=rate, ttft_s=(t_first - t0) if t_first else None,
                    server_timings={k: timings.get(k) for k in ('predicted_per_second', 'draft_n', 'draft_n_accepted', 'predicted_n')} if timings else None,
                    text_head=''.join(text_out)[:160])

    def config_round(self, name, cfg, rnd):
        labels = [f'{name}-r{rnd}-{pid}' for pid, _ in PROMPTS]
        while not all(l in self.done for l in labels):
            self.quiet()                                   # not holding the GPU here
            try:
                self.one_pass(name, cfg, labels, rnd)
            except Requeue as error:
                self.event(kind='requeue', config=name, round=rnd, reason=str(error) or f'waited over {HELD_WAIT_S} s with the GPU held')

    def one_pass(self, name, cfg, labels, rnd):
        pending = []
        try:
            self.start(name, cfg)
            self.request(name, cfg, 'Say hello in five words.', 16)   # warm-up, not counted
            for (pid, text), label in zip(PROMPTS, labels):
                if not self.load_gate and label in self.done:
                    continue
                for attempt in range(1, 5):
                    waited, pre = self.quiet()
                    pids = [] if self.load_gate else engine_pids(name)
                    b0, o0, l0, t0 = busy_jiffies(), proc_jiffies(pids), os.getloadavg()[0], time.time()
                    res = self.request(name, cfg, text, 256)
                    wall = time.time() - t0
                    foreign = max(0.0, ((busy_jiffies() - b0) - (proc_jiffies(pids) - o0)) / HZ / wall)
                    l1 = os.getloadavg()[0]
                    if self.load_gate:
                        self.check_gpu()
                    ok = ((l0 < 3 and l1 < 3) if self.load_gate else foreign < 1.0) and res['decode_tok_s'] is not None
                    vram = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
                    record = dict(kind='request', label=label, config=name, round=rnd, prompt=pid, attempt=attempt, waited_s=waited,
                               pre_busy_cores=None if self.load_gate else pre, load_start=l0, load_end=l1, gate='load' if self.load_gate else 'foreign', wall_s=round(wall, 2), foreign_cores=None if self.load_gate else round(foreign, 2),
                               vram_mib=vram, accepted=ok, **res)
                    if self.load_gate:
                        self.event(**{**record, 'kind':'request_pending'})
                        if not ok:
                            raise Requeue('request failed the load or result gate; repeat the full prompt pass')
                        pending.append(record)
                        break
                    self.event(**record)
                    if ok:
                        self.done.add(label)
                        break
            for record in pending:
                self.event(**record)
                self.done.add(record['label'])
        finally:
            self.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('deadline'); ap.add_argument('--only', nargs='*'); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--load-gate', action='store_true', help='sandbox: require 1-minute load below 3 before and after each request')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    h, m = map(int, a.deadline.split(':')); t = time.gmtime()
    end = calendar.timegm((t.tm_year, t.tm_mon, t.tm_mday, h, m, 0)); end = end if end > time.time() else end + 86400
    run = Run(out, end, a.load_gate)
    names = a.only or list(CONFIGS)
    if a.load_gate and any('gateway_model' in CONFIGS[name] for name in names):
        ap.error('--load-gate requires --only with local llama.cpp configurations')
    run.event(kind='start', configs=names, rounds=a.rounds, prompts=[p for p, _ in PROMPTS], pid=os.getpid())
    def on_term(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, on_term)
    try:
        for rnd in range(1, a.rounds + 1):
            k = (rnd - 1) % len(names)
            for name in names[k:] + names[:k]:          # rotate the order each round
                run.config_round(name, CONFIGS[name], rnd)
        run.event(kind='all_complete')
    except (TimeoutError, KeyboardInterrupt) as e:
        run.event(kind='stop', reason=str(e) or type(e).__name__)
    finally:
        run.stop()


if __name__ == '__main__':
    main()
