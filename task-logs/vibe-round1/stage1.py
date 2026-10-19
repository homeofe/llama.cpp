#!/usr/bin/env python3
"""Vibe round 1, stage 1: Strata's round measured the same way as ours.

Subcommands:
  bench        start the deployed Strata on 18100, warm up, N rounds of the 8 stage_b prompts
  nsys-strata  one steady code request on Strata under Nsight Systems (CUDA + OS runtime)
  nsys-ours    the same request on this worktree's llama-server, q-r8-slots, graphs disabled
Every run holds the GPU at the gateway, releases it in every exit path, and applies the
quiet gate (1-minute load below 3, no foreign process above one core).
The gateway API key is read but never printed or written.
"""
import argparse, hashlib, json, os, secrets, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, '/opt/llm-gateway/data/moe-quant/qwen38')
import stage_b  # noqa: E402  (PROMPTS, the deployed harness's own module)

ROOT = Path('/opt/llm-gateway/data/build/llama-vibe')
OUT = ROOT / 'task-logs/vibe-round1'
SRV = Path('/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb')
CFG_DEPLOYED = SRV / 'strata-iq3_s.json'
PY = '/opt/llm-gateway/data/venvs/strata/bin/python'
GW = 'http://127.0.0.1:8790'
KEY = Path('/opt/llm-gateway/data/api-key').read_text().strip()
PORT = 18100
MYKEY = 'vibe-' + secrets.token_hex(16)
HZ = os.sysconf('SC_CLK_TCK')
PROMPTS = stage_b.PROMPTS


def gw(method, path, payload=None, timeout=60):
    r = urllib.request.Request(GW + path, method=method,
                               data=json.dumps(payload).encode() if payload is not None else None,
                               headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return f.read().decode()


class Hold:
    def __init__(self, log):
        self.held = False
        self.log = log
        self.t = 0.0
        self.stop_flag = False
        self.thread = None

    def acquire(self):
        gw('POST', '/api/v1/gpu/free', {})
        gw('POST', '/api/v1/gpu/hold', {'reason': 'vibe: strata parity', 'ttlS': 900})
        self.held = True
        self.t = time.monotonic()
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True)
        busy = {int(l.split(',')[0]): int(l.split(',')[1]) for l in apps.splitlines() if l.strip()}
        if any(m > 300 for m in busy.values()):
            raise RuntimeError('GPU in use: ' + str(busy))
        self.thread = threading.Thread(target=self._renew, daemon=True)
        self.thread.start()

    def _renew(self):
        while not self.stop_flag:
            time.sleep(30)
            if self.stop_flag or not self.held:
                return
            try:
                gw('POST', '/api/v1/gpu/hold', {'reason': 'vibe: strata parity', 'ttlS': 900})
            except Exception as e:
                self.log(dict(kind='hold_error', error=str(e)))

    def release(self):
        self.stop_flag = True
        if self.held:
            try:
                gw('DELETE', '/api/v1/gpu/hold')
            except Exception:
                pass
            self.held = False


def load_ok():
    return os.getloadavg()[0] < 3


def wait_quiet(end, hold, timeout=900):
    t0 = time.time()
    while time.time() < end:
        if load_ok():
            return round(time.time() - t0)
        if time.time() - t0 > timeout:
            raise TimeoutError('no quiet window')
        time.sleep(10)
    raise TimeoutError('deadline')


def busy_jiffies():
    v = list(map(int, open('/proc/stat').readline().split()[1:]))
    return sum(v[:8]) - v[3] - v[4]


def proc_jiffies(pids):
    t = 0
    for p in pids:
        try:
            f = open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()
            t += int(f[11]) + int(f[12])
        except (OSError, IndexError):
            pass
    return t


def own_pids(marker):
    pids = []
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            cmd = open(f'/proc/{d}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if marker in cmd and 'stage1.py' not in cmd:
            pids.append(int(d))
    return pids


def http_json(url, key=None, timeout=10):
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + key} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode())


def request(url, key, text, max_tokens, chunk_times=None, tag=''):
    body = dict(messages=[{'role': 'user', 'content': text}], max_tokens=max_tokens, temperature=0, stream=True,
                stream_options={'include_usage': True}, chat_template_kwargs={'enable_thinking': False},
                model='qwen3.8-flash-next-iq3_s')
    r = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json',
                                        'Authorization': 'Bearer ' + key})
    t_first = t_last = None
    pieces = 0
    usage = timings = None
    text_out = []
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=1800) as f:
        for raw in f:
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
                    if chunk_times is not None:
                        chunk_times.append((round((now - t0) * 1000, 1), len(c)))
    txt = ''.join(text_out)
    n = (usage or {}).get('completion_tokens') or pieces
    return dict(tokens=n, pieces=pieces, decode_tok_s=(n - 1) / (t_last - t_first) if t_first and t_last and n > 1 else None,
                ttft_s=(t_first - t0) if t_first else None, wall_s=time.time() - t0,
                sha256=hashlib.sha256(txt.encode()).hexdigest(), text_head=txt[:120],
                timings=timings, usage=usage)


class Sampler:
    """Per-thread CPU jiffy deltas of one process during a request (no perf events needed)."""

    def __init__(self, pid):
        self.pid = pid
        self.samples = []
        self.running = False

    def _read(self):
        out = {}
        base = f'/proc/{self.pid}/task'
        try:
            for t in os.listdir(base):
                try:
                    f = open(f'{base}/{t}/stat').read().rsplit(')', 1)[1].split()
                    out[int(t)] = int(f[11]) + int(f[12])
                except (OSError, IndexError):
                    pass
        except OSError:
            pass
        return out

    def start(self):
        self.prev = self._read()
        self.t0 = time.monotonic()
        self.running = True

        def run():
            while self.running:
                time.sleep(0.05)
                cur = self._read()
                if cur:
                    self.samples.append(sum(max(0, cur.get(t, 0) - v) for t, v in self.prev.items()))
                    self.prev = cur

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        return dict(pid=self.pid, seconds=round(time.monotonic() - self.t0, 2),
                    n_samples=len(self.samples),
                    mean_threads=sum(self.samples) / max(1, len(self.samples)) * (0.05 * HZ),
                    peak_threads=max(self.samples or [0]) * (0.05 * HZ))


def last_history(metrics):
    hist = metrics.get('history') if isinstance(metrics, dict) else None
    if isinstance(hist, list) and hist:
        return hist[-1]
    if isinstance(hist, dict) and hist:
        return hist[next(reversed(list(hist)))]
    return metrics


def write_config(out):
    cfg = json.loads(CFG_DEPLOYED.read_text())
    cfg['log'] = str(out / 'engine.log')
    cfg['api_key'] = MYKEY
    (out / 'config.json').write_text(json.dumps(cfg))
    return out / 'config.json'


def start_strata(out, extra_argv=(), extra_env=None, stdout_name='server.log'):
    cfg_path = write_config(out)
    (out / 'ws').mkdir(exist_ok=True)
    env = dict(os.environ, STRATA_API_KEY=MYKEY, STRATA_BF16_TC='1', STRATA_NO_LARGEPAGES='1',
               STRATA_SELECT_SIMT='1')
    env.update(extra_env or {})
    cmd = [PY, str(SRV / 'serve/server.py'), '--engine', 'strata', '--config', str(cfg_path),
           '--host', '127.0.0.1', '--port', str(PORT), '--engine-background',
           '--workspace-dir', str(out / 'ws'), '--idle-unload', '0'] + list(extra_argv)
    log = (out / stdout_name).open('a')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           env=env, start_new_session=True)
    log.close()
    return proc


def wait_ready(url, key, proc, end, log, timeout=600):
    """Poll /health; Strata says "loaded", llama-server just answers 200."""
    t0 = time.time()
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}')
        try:
            j = http_json(url + '/health', key=key or None)
            if j.get('loaded') is False:
                if time.time() - t0 > timeout:
                    raise TimeoutError('engine not loaded in time')
                time.sleep(3)
                continue
            log(kind='ready', load_s=round(time.time() - t0, 1))
            return time.time() - t0
        except Exception:
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError('engine not loaded in time')
        time.sleep(3)
    raise TimeoutError('deadline while loading')


class Ledger:
    def __init__(self, out):
        self.path = out / 'events.jsonl'

    def __call__(self, **kw):
        kw['utc'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        with self.path.open('a') as f:
            f.write(json.dumps(kw) + '\n')
        print(json.dumps(kw), flush=True)


def gate_request(url, key, text, label, log, end, sampler_pid=None, max_tokens=256):
    """Quiet gate before/after, foreign-core accounting during."""
    waited = wait_quiet(end, None)
    l0 = os.getloadavg()[0]
    pids = own_pids('runtimes/strata') + own_pids('llama-server')
    b0, o0 = busy_jiffies(), proc_jiffies(pids)
    chunk_times = []
    sampler = Sampler(sampler_pid) if sampler_pid else None
    if sampler:
        sampler.start()
    res = request(url, key, text, max_tokens, chunk_times=chunk_times)
    if sampler:
        res['thread_sample'] = sampler.stop()
    wall = res.pop('wall_s')
    foreign = max(0.0, ((busy_jiffies() - b0) - (proc_jiffies(pids) - o0)) / HZ / wall)
    l1 = os.getloadavg()[0]
    res.update(label=label, waited_s=waited, load_start=round(l0, 2), load_end=round(l1, 2),
               foreign_cores=round(foreign, 2), wall_s=round(wall, 2), chunk_times=chunk_times)
    ok = l0 < 3 and l1 < 3 and foreign < 1.0 and res['decode_tok_s'] is not None
    res['accepted'] = ok
    return res


def cmd_bench(a, out, log):
    end = time.time() + 3600
    hold = Hold(log)
    proc = None
    try:
        wait_quiet(end, None, timeout=1800)
        hold.acquire()
        log(kind='hold_acquired')
        proc = start_strata(out)
        log(kind='server_pid', pid=proc.pid)
        url = f'http://127.0.0.1:{PORT}'
        load_s = wait_ready(url, MYKEY, proc, end, log)
        warm = request(url, MYKEY, 'Say hello in five words.', 16)
        log(kind='warmup', tokens=warm['tokens'])
        engine_pid = None
        for p in own_pids('engine/strata'):
            engine_pid = p
        sampler_round = a.sampler_round
        for rnd in range(1, a.rounds + 1):
            for idx, (pid, text) in enumerate(PROMPTS):
                sample_pid = engine_pid if (rnd == sampler_round and idx == a.sampler_prompt) else None
                label = f'strata-r{rnd}-{pid}'
                while True:
                    res = gate_request(url, MYKEY, text, label, log, end, sampler_pid=sample_pid)
                    log(kind='request', **res)
                    if res['accepted']:
                        break
                    log(kind='gate_failed', label=res['label'])
                    wait_quiet(end, None, timeout=600)
                metrics = http_json(url + '/metrics', key=MYKEY)
                log(kind='metrics', label=label, metrics=last_history(metrics))
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for p in own_pids('engine/strata'):        # --engine-background can orphan the engine
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(3)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                      text=True).split()
        hold.release()
        log(kind='hold_released', gpu_apps_left=apps)


def run_nsys(out, log, name, cmd, env, key, prompt_text, marker, end, port=PORT):
    hold = Hold(log)
    proc = None
    try:
        wait_quiet(end, None, timeout=1800)
        hold.acquire()
        log(kind='hold_acquired', capture=name)
        argv = ['nsys', 'profile', '--trace=cuda,osrt,nvtx', '--sample=none', '--cpuctxsw=none',
                '--kill=none', '--force-overwrite=true', '--delay=5', '--duration=900',
                '-o', str(out / name)] + cmd
        logf = (out / (name + '.nsys.log')).open('a')
        proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, start_new_session=True)
        logf.close()
        log(kind='nsys_pid', pid=proc.pid, capture=name)
        url = f'http://127.0.0.1:{port}'
        wait_ready(url, key, proc, end, log)
        warm = request(url, key, 'Say hello in five words.', 16)
        log(kind='warmup', capture=name, tokens=warm['tokens'])
        engine_pid = None
        for p in own_pids(marker):
            engine_pid = p
        res = gate_request(url, key, prompt_text, name + '-steady', log, end, sampler_pid=engine_pid)
        log(kind='request', capture=name, **res)
        try:
            metrics = http_json(url + '/metrics', key=key or None)
            log(kind='metrics', capture=name, metrics=last_history(metrics))
        except Exception as e:
            log(kind='metrics_error', capture=name, error=repr(e))
        time.sleep(2)
        return res
    finally:
        # graceful first: the engine's own /engine/stop endpoint lets the server reap it,
        # so the injected process tree exits cleanly and nsys can finalize its report
        try:
            req = urllib.request.Request(url + '/engine/stop', method='POST', data=b'',
                                         headers={'Authorization': 'Bearer ' + key} if key else {})
            urllib.request.urlopen(req, timeout=30).read()
            log(kind='engine_stop_sent', capture=name)
        except Exception as e:
            log(kind='engine_stop_error', capture=name, error=repr(e))
        deadline = time.time() + 60
        while time.time() < deadline and own_pids(marker):
            time.sleep(2)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            pids = own_pids(marker) + own_pids('serve/server.py')
            for p in pids:
                try:
                    os.kill(p, sig)
                except ProcessLookupError:
                    pass
            deadline = time.time() + (240 if sig == signal.SIGINT else 120)
            while own_pids(marker) and time.time() < deadline:
                time.sleep(1)
            if not own_pids(marker):
                break
        time.sleep(3)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                      text=True).split()
        hold.release()
        log(kind='hold_released', capture=name, gpu_apps_left=apps)
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log(kind='nsys_exit', capture=name, code=proc.returncode if proc else None)


def cmd_nsys_strata(a, out, log):
    end = time.time() + 3600
    # start the server under nsys; the engine child inherits the injection
    cfg_path = write_config(out)
    cmd_prefix = [PY, str(SRV / 'serve/server.py'), '--engine', 'strata', '--config', str(cfg_path),
                  '--host', '127.0.0.1', '--port', str(PORT), '--engine-background',
                  '--workspace-dir', str(out / 'ws'), '--idle-unload', '0']
    env = dict(os.environ, STRATA_API_KEY=MYKEY, STRATA_BF16_TC='1', STRATA_NO_LARGEPAGES='1',
               STRATA_SELECT_SIMT='1')
    run_nsys(out, log, 'strata-code1', cmd_prefix, env, MYKEY, PROMPTS[a.prompt][1], 'engine/strata', end)
    if not a.no_second:
        en_text = next(p for i, p in enumerate(PROMPTS) if p[0] == 'en-1')
        run_nsys(out, log, 'strata-en1', cmd_prefix, env, MYKEY, en_text, 'engine/strata', end)


def cmd_strata_shim(a, out, log):
    end = time.time() + 3600
    cfg = json.loads(CFG_DEPLOYED.read_text())
    cfg['log'] = str(out / 'engine.log')
    cfg['api_key'] = MYKEY
    real = cfg['exe']
    cfg['exe'] = str(ROOT / 'task-logs/vibe-round1/engine-shim.sh')
    (out / 'config.json').write_text(json.dumps(cfg))
    (out / 'real-exe.txt').write_text(real)
    hold = Hold(log)
    proc = None
    try:
        wait_quiet(end, None, timeout=1800)
        hold.acquire()
        log(kind='hold_acquired', capture='strata-shim')
        env = dict(os.environ, STRATA_API_KEY=MYKEY, STRATA_BF16_TC='1', STRATA_NO_LARGEPAGES='1',
                   STRATA_SELECT_SIMT='1', NSYS_OUT=str(out / 'strata-shim'),
                   STRATA_REAL_EXE=real)
        cfg_path = out / 'config.json'
        (out / 'ws').mkdir(exist_ok=True)
        cmd = [PY, str(SRV / 'serve/server.py'), '--engine', 'strata', '--config', str(cfg_path),
               '--host', '127.0.0.1', '--port', str(PORT), '--engine-background',
               '--workspace-dir', str(out / 'ws'), '--idle-unload', '0']
        logf = (out / 'strata-shim.server.log').open('a')
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, start_new_session=True)
        logf.close()
        log(kind='server_pid', pid=proc.pid, capture='strata-shim')
        url = f'http://127.0.0.1:{PORT}'
        wait_ready(url, MYKEY, proc, end, log, timeout=420)
        warm = request(url, MYKEY, 'Say hello in five words.', 16)
        log(kind='warmup', capture='strata-shim', tokens=warm['tokens'])
        engine_pid = None
        for p in own_pids('engine/strata'):
            engine_pid = p
        res = gate_request(url, MYKEY, PROMPTS[a.prompt][1], 'strata-shim-steady', log, end,
                           sampler_pid=engine_pid)
        log(kind='request', capture='strata-shim', **res)
        try:
            metrics = http_json(url + '/metrics', key=MYKEY)
            log(kind='metrics', capture='strata-shim', metrics=last_history(metrics))
        except Exception as e:
            log(kind='metrics_error', capture='strata-shim', error=repr(e))
    finally:
        # the server's SIGINT shutdown sends QUIT to the engine; nsys then writes its report
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        deadline = time.time() + 360
        report = None
        while time.time() < deadline:
            for suffix in ('.nsys-rep', '.qdstrm'):
                f = out / ('strata-shim' + suffix)
                if f.exists() and f.stat().st_size > 100000 and time.time() - f.stat().st_mtime > 10:
                    report = str(f)
                    break
            if report:
                break
            time.sleep(5)
        for p in own_pids('engine/strata') + own_pids('serve/server.py'):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        hold.release()
        log(kind='hold_released', capture='strata-shim', report=report)


def cmd_nsys_ours(a, out, log):
    end = time.time() + 3600
    cfg = stage_b.CONFIGS['q-r8-slots']
    env = dict(os.environ, **cfg['env'], GGML_CUDA_DISABLE_GRAPHS='1', GGML_CUDA_BF16_AS_F16='1')
    ours = str(ROOT / 'build-cuda/bin/llama-server')
    cmd_prefix = [(ours if c == stage_b.L5 else c) for c in cfg['cmd']]
    run_nsys(out, log, 'ours-code1', cmd_prefix, env, '', PROMPTS[a.prompt][1], 'llama-server', end,
             port=int(cfg['cmd'][cfg['cmd'].index('--port') + 1]))
    if not a.no_second:
        en_text = next(p for i, p in enumerate(PROMPTS) if p[0] == 'en-1')
        run_nsys(out, log, 'ours-en1', cmd_prefix, env, '', en_text, 'llama-server', end,
                 port=int(cfg['cmd'][cfg['cmd'].index('--port') + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['bench', 'nsys-strata', 'nsys-ours', 'strata-shim'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--prompt', type=int, default=0, help='index into stage_b.PROMPTS for the profiled request')
    ap.add_argument('--sampler-round', type=int, default=2)
    ap.add_argument('--sampler-prompt', type=int, default=0)
    ap.add_argument('--no-second', action='store_true')
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT / {'bench': 'strata-bench', 'nsys-strata': 'strata-nsys',
                                          'nsys-ours': 'ours-nsys', 'strata-shim': 'strata-shim'}[a.command]
    out.mkdir(parents=True, exist_ok=True)
    log = Ledger(out)
    log(kind='start', command=a.command, out=str(out))
    try:
        if a.command == 'bench':
            cmd_bench(a, out, log)
        elif a.command == 'nsys-strata':
            cmd_nsys_strata(a, out, log)
        elif a.command == 'strata-shim':
            cmd_strata_shim(a, out, log)
        else:
            cmd_nsys_ours(a, out, log)
        log(kind='complete')
    except Exception as e:
        log(kind='error', error=repr(e))
        raise


if __name__ == '__main__':
    main()
