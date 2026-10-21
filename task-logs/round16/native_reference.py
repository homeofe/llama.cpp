#!/usr/bin/env python3
"""Run the actual Strata configuration and llama baseline, with paired prompt reuse.

The first request is novel to this process, not a promise of an empty prefix or
expert cache. Any contaminated block is discarded and replayed from a fresh process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time
import urllib.request

from run_drafts import Run, Contaminated, HERE, bench, configs

STRATA = Path('/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb')
PYTHON = '/opt/llm-gateway/data/venvs/strata/bin/python'


def identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class Reference(Run):
    def __init__(self, out, end):
        super().__init__(out, end)
        self.native = False
        self.engine = None
        self.native_attempt = 0
        self.auth = secrets.token_hex(24)
        self.url = f'http://127.0.0.1:{bench.PORT}'

    def owned(self):
        pids = [self.child.pid] if self.child else []
        if self.engine and identity(self.engine[0]) == self.engine[1]:
            pids.append(self.engine[0])
        return pids

    def api(self, path, method='GET'):
        req = urllib.request.Request(self.url + path, method=method,
                                     data=b'{}' if method == 'POST' else None,
                                     headers={'Authorization': 'Bearer ' + self.auth, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    def quiet(self):
        start = time.time()
        while time.time() < self.end:
            self.renew()
            b0, o0 = bench.busy_jiffies(), bench.proc_jiffies(self.owned())
            time.sleep(2)
            foreign = max(0, (bench.busy_jiffies()-b0-bench.proc_jiffies(self.owned())+o0)/bench.HZ/2)
            if os.getloadavg()[0] < 2 and foreign < 1:
                return time.time()-start, foreign
        raise TimeoutError('quiet-window deadline')

    def start(self, name, cfg):
        self.native = name == 'strata'
        if self.native:
            self.native_attempt += 1
        self.url = f'http://127.0.0.1:{18100 if self.native else bench.PORT}'
        if not self.native:
            return super().start(name, cfg)
        self.held = True
        self.renew(force=True)
        bench.gw('POST', '/api/v1/gpu/free', {})
        before = self.gpu_apps()
        if any(mib > 300 for mib in before.values()):
            raise RuntimeError('GPU already in use')
        native_cfg = json.loads((STRATA/'strata-iq3_s.json').read_text())
        native_cfg.pop('api_key', None)
        native_cfg['log'] = str(self.out/'strata.engine.log')
        cfg_path = self.out/'strata-config.json'
        cfg_path.write_text(json.dumps(native_cfg, indent=2)+'\n')
        env = dict(os.environ, STRATA_API_KEY=self.auth, STRATA_BF16_TC='1',
                   STRATA_NO_LARGEPAGES='1', STRATA_SELECT_SIMT='1')
        env.pop('LD_LIBRARY_PATH', None)
        cmd = [PYTHON, str(STRATA/'serve/server.py'), '--engine', 'strata',
               '--config', str(cfg_path), '--host', '127.0.0.1', '--port', '18100',
               '--engine-background', '--workspace-dir', str(self.out/'workspace'/f'attempt-{self.native_attempt}'), '--idle-unload', '0']
        with (self.out/'strata.server.txt').open('a') as log:
            self.child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = min(self.end, time.time()+300)
        while time.time() < deadline:
            if self.child.poll() is not None:
                raise RuntimeError(f'Strata wrapper exited {self.child.returncode}')
            for pid in set(self.gpu_apps())-set(before):
                try:
                    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                    if argv[0].decode() == native_cfg['exe']:
                        self.engine = (pid, identity(pid))
                except OSError:
                    pass
            try:
                if self.api('/health').get('loaded') is True and self.engine:
                    return
            except Exception:
                pass
            self.renew()
            time.sleep(2)
        raise TimeoutError('Strata startup deadline')

    def stop(self):
        try:
            if self.native and self.child and self.child.poll() is None:
                try:
                    self.api('/engine/stop', 'POST')
                except Exception as error:
                    self.event(kind='engine_stop_error', error=str(error))
            if self.engine and identity(self.engine[0]) == self.engine[1]:
                os.kill(self.engine[0], signal.SIGTERM)
                deadline = time.time()+15
                while identity(self.engine[0]) == self.engine[1] and time.time() < deadline:
                    time.sleep(.2)
                if identity(self.engine[0]) == self.engine[1]:
                    os.kill(self.engine[0], signal.SIGKILL)
        finally:
            self.engine = None
            super().stop()

    def request(self, prompt, tag, tokens=256):
        body = dict(model='qwen3.8-flash-next-iq3_s', messages=[dict(role='user', content=prompt)],
                    max_tokens=tokens, temperature=0, seed=1, stream=True,
                    stream_options=dict(include_usage=True), cache_prompt=True,
                    chat_template_kwargs=dict(enable_thinking=False))
        req = urllib.request.Request(self.url+'/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json', 'Authorization':'Bearer '+self.auth})
        start = time.monotonic()
        first = last = None
        chunks = []
        usage = timings = None
        finished = False
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    finished = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                usage = event.get('usage') or usage
                timings = event.get('timings') or timings
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    piece = delta.get('content') or delta.get('reasoning_content')
                    if piece:
                        last = time.monotonic()
                        first = first or last
                        chunks.append(piece)
        wall = time.monotonic()-start
        assert finished and usage and chunks, 'incomplete response'
        answer = ''.join(chunks)
        (self.out/(tag+'.txt')).write_text(answer)
        return dict(tokens=usage['completion_tokens'], usage=usage, timings=timings,
                    sha256=hashlib.sha256(answer.encode()).hexdigest(), wall_s=wall,
                    ttft_s=first-start, decode_tok_s=(usage['completion_tokens']-1)/(last-first))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engines', nargs='+', choices=['strata', 'base', 'strata-gpu-global', 'strata-reference'], default=['strata','base'])
    ap.add_argument('--minutes', type=int, default=40)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = HERE/a.out
    out.mkdir(exist_ok=False)
    cfgs = configs()
    (out/'llama-configs.json').write_text(json.dumps({n:cfgs[n] for n in a.engines if n!='strata'}, indent=2)+'\n')
    run = Reference(out, time.time()+a.minutes*60)
    def stop(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    try:
        for name in a.engines:
            for attempt in range(3):
                try:
                    run.quiet()
                    run.event(kind='start_config', config=name, block_attempt=attempt)
                    run.start(name, cfgs.get(name))
                    prefix = f'{name}-b{attempt}'
                    run.request('Say hello in five words.', prefix+'-warmup', 16)
                    for prompt_id, prompt in bench.PROMPTS:
                        for phase in ('novel','repeat'):
                            waited, _ = run.quiet()
                            load = os.getloadavg()[0]
                            b0, o0 = bench.busy_jiffies(), bench.proc_jiffies(run.owned())
                            result = run.request(prompt, prefix+'-'+prompt_id+'-'+phase)
                            foreign = max(0,(bench.busy_jiffies()-b0-bench.proc_jiffies(run.owned())+o0)/bench.HZ/result['wall_s'])
                            run.event(kind='request', config=name, block_attempt=attempt, prompt=prompt_id, phase=phase,
                                      load_start=load, load_end=os.getloadavg()[0], waited_s=waited, foreign_cores=foreign,
                                      accepted=load<2 and foreign<1, gpu=run.gpu_apps(), **result)
                            if foreign >= 1 or load >= 2:
                                raise Contaminated('reset whole process after foreign load')
                            if name == 'strata':
                                metrics = run.api('/metrics')
                                hist = metrics.get('history') or []
                                last = hist[-1] if isinstance(hist,list) and hist else None
                                run.event(kind='metrics', prompt=prompt_id, phase=phase, block_attempt=attempt, metrics=last)
                    run.event(kind='block_complete', config=name, block_attempt=attempt)
                    break
                except Contaminated as error:
                    run.event(kind='block_discarded', config=name, block_attempt=attempt, error=str(error))
                    if attempt == 2:
                        raise
                finally:
                    run.stop()
        run.event(kind='complete')
    finally:
        run.stop()
        run.event(kind='released', gpu=run.gpu_apps())


if __name__ == '__main__':
    main()
