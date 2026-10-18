#!/usr/bin/env python3
"""Run the existing panels through an isolated gateway and a per-request quiet gate."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

import bench

ROOT = Path(__file__).resolve().parents[2]
HOME_BENCH = Path('/opt/llm-gateway/data/bench-home-keep75ud')
OUT = ROOT / 'task-logs/round11/panels'
MODEL_ID = 'ling-round11'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    bench.configure()
    cfg = bench.stage.CONFIGS[args.config]
    OUT.mkdir(exist_ok=True)
    if (OUT / 'model-entry.json').exists():
        raise RuntimeError('panel directory already used; archive it before a retry')
    settings_path = HOME_BENCH / 'data/settings.json'
    original = settings_path.read_bytes()
    backup = HOME_BENCH / 'data/settings.before-round11.json'
    if not backup.exists():
        backup.write_bytes(original)
    settings = json.loads(original)
    flags = cfg['cmd'][1:]
    model = Path(flags[flags.index('-m') + 1])
    model_dir = HOME_BENCH / 'models/ling-round11'
    model_dir.mkdir(exist_ok=True)
    for shard in sorted(model.parent.glob('*.gguf')):
        link = model_dir / shard.name
        if not link.exists():
            os.link(shard, link)
        assert link.stat().st_ino == shard.stat().st_ino
    extra = []
    i = 0
    while i < len(flags):
        if flags[i] in ['-m', '--host', '--port']:
            i += 2
        else:
            extra.append(flags[i])
            i += 1
    entry = dict(id=MODEL_ID, label='Ling 3.0 Flash IQ4_XS (' + args.config + ')', binary=cfg['cmd'][0],
                 args=extra, env=cfg['env'], sampling=dict(temperature=0.6, top_p=0.95, top_k=20), reasoning='on',
                 capabilities=dict(vision=False, tools=True, reasoning=True, context=int(flags[flags.index('-c') + 1])))
    settings['models']['ling-round11/Ling-3.0-flash-IQ4_XS'] = entry
    assert settings['keepLoaded'] == []
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')
    (OUT / 'model-entry.json').write_text(json.dumps(entry, indent=2) + '\n')
    run = bench.Run(OUT, 1792336440)  # 15:14 UTC: leave ten minutes for final checks.
    gateway = None
    panel = None
    server = None
    key = Path(settings['apiKeyFile']).read_text().strip()
    env = {**os.environ, 'LLM_GATEWAY_HOME': str(HOME_BENCH), 'LLM_GATEWAY_KEY_FILE': settings['apiKeyFile'],
           'CUDA_CACHE_PATH': str(ROOT / 'task-logs/round11/cuda-cache'),
           'npm_config_cache': str(HOME_BENCH / 'npm-cache'), 'TMPDIR': str(HOME_BENCH / 'tmp')}
    Path(env['TMPDIR']).mkdir(exist_ok=True)

    def api(method, path, body=None):
        request = urllib.request.Request('http://127.0.0.1:8791' + path, method=method,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
            data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def own_pids():
        result = [os.getpid()]
        if gateway is None:
            return result
        result.append(gateway.pid)
        for d in Path('/proc').iterdir():
            if not d.name.isdigit():
                continue
            try:
                fields = (d / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[1]) == gateway.pid:
                    result.append(int(d.name))
            except (OSError, IndexError, ValueError):
                pass
        return result

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            is_chat = self.path.startswith('/v1/chat/completions')
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ['host', 'connection', 'content-length']}
            for attempt in range(1, 4):
                if is_chat:
                    run.quiet()
                    foreign_gpu = {p: m for p, m in run.gpu_apps().items() if p not in own_pids() and m > 300}
                    if foreign_gpu:
                        raise RuntimeError('foreign GPU allocation ' + str(foreign_gpu))
                b0, o0, t0 = bench.stage.busy_jiffies(), bench.stage.proc_jiffies(own_pids()), time.monotonic()
                request = urllib.request.Request('http://127.0.0.1:8791' + self.path, data=body or None, method=self.command, headers=headers)
                try:
                    response = urllib.request.urlopen(request, timeout=min(1800, max(1, run.end - time.time())))
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    payload = response.read()
                    status = int(response.status or 500)
                wall = time.monotonic() - t0
                if not is_chat:
                    break
                foreign = max(0, bench.stage.busy_jiffies() - b0 - bench.stage.proc_jiffies(own_pids()) + o0) / bench.stage.HZ / wall
                run.event(kind='panel_request', path=self.path, status=status, wall_s=wall,
                    foreign_cores=foreign, quiet=foreign < 1.0, attempt=attempt, selected=foreign < 1.0,
                    request_sha256=hashlib.sha256(body).hexdigest(), response_sha256=hashlib.sha256(payload).hexdigest())
                with (OUT / 'responses.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(status=status, attempt=attempt, selected=foreign < 1.0,
                        request=json.loads(body), response=json.loads(payload))) + '\n')
                if foreign < 1.0:
                    break
            else:
                status = 503
                payload = json.dumps(dict(error=dict(message='Foreign CPU gate failed three attempts.'))).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    try:
        run.quiet()
        bench.stage.gw('POST', '/api/v1/gpu/free', {})
        run.held = True
        run.renew(force=True)
        if any(m > 300 for m in run.gpu_apps().values()):
            raise RuntimeError('GPU already in use')
        with (OUT / 'gateway.log').open('w') as stream:
            gateway = subprocess.Popen(['node', '/opt/llm-gateway/apps/service/dist/main.js'], env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(60):
            if gateway.poll() is not None:
                raise RuntimeError('second gateway exited')
            try:
                models = api('GET', '/api/v1/models')['models']
                assert any(m['id'] == MODEL_ID for m in models), 'Ling was not discovered'
                (OUT / 'discovered-model.json').write_text(json.dumps(next(m for m in models if m['id'] == MODEL_ID), indent=2) + '\n')
                break
            except urllib.error.URLError:
                time.sleep(1)
        else:
            raise TimeoutError('gateway did not start')
        server = ThreadingHTTPServer(('127.0.0.1', 18791), Proxy)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        command = ['npm', 'run', '--silent', 'bench', '--', '--label', 'card-round11-2026-10-18',
                   '--card', '--hard', '--agent', '--runs', '3', '--no-speed', MODEL_ID]
        (OUT / 'command.json').write_text(json.dumps(dict(command=command, gateway_port=8791, gate_proxy_port=18791,
            original_context=int(flags[flags.index('-c') + 1]), long_panel_context_limitation=False,
            retry_policy='Up to three identical requests, selected only by foreign CPU < 1 core; all attempts archived.'), indent=2) + '\n')
        with (OUT / 'bench.log').open('w') as stream:
            panel = subprocess.Popen(command, cwd='/opt/llm-gateway', env={**env, 'LLM_GATEWAY_URL': 'http://127.0.0.1:18791'},
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        run.event(kind='panel_started', gateway_pid=gateway.pid, bench_pid=panel.pid, config=args.config)
        while panel.poll() is None:
            run.renew()
            if not (OUT / 'actual-engine.json').exists():
                for pid in own_pids():
                    try:
                        argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
                        if not argv or argv[0] != cfg['cmd'][0]:
                            continue
                        environ = dict(x.split('=', 1) for x in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in x)
                        libraries = sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                            if '/libllama' in line or '/libggml' in line})
                        (OUT / 'actual-engine.json').write_text(json.dumps(dict(pid=pid, argv=argv,
                            env={k: v for k, v in environ.items() if k.startswith(('GGML_', 'LD_', 'OMP_', 'CUDA_'))},
                            libraries=libraries), indent=2) + '\n')
                    except FileNotFoundError:
                        pass
            if time.time() > run.end:
                raise TimeoutError('panel deadline')
            time.sleep(2)
        run.event(kind='panel_exit', exit=panel.returncode)
        if panel.returncode:
            raise RuntimeError('panel process failed')
    finally:
        try:
            if panel is not None and panel.poll() is None:
                os.killpg(panel.pid, signal.SIGTERM)
                panel.wait(timeout=30)
            if gateway is not None and gateway.poll() is None:
                try:
                    api('POST', '/api/v1/gpu/free', {})
                except Exception as error:
                    run.event(kind='gateway_unload_error', error=str(error))
                gateway.terminate()
                try:
                    gateway.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(gateway.pid, signal.SIGKILL)
                    gateway.wait(timeout=10)
        finally:
            if server is not None:
                server.shutdown()
            run.stop()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
