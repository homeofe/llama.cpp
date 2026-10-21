#!/usr/bin/env python3
"""Run existing code/tools/German/format checks on one directly served candidate."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from run_drafts import Run, bench, configs, HERE


def main():
    name, folder = sys.argv[1:]
    out = HERE / folder
    out.mkdir(exist_ok=False)
    run = Run(out, time.time()+1200)
    cfg = configs()[name]
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    proxy = child = None
    def stop(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            try:
                run.quiet()
                b0, o0, t0 = bench.busy_jiffies(), bench.proc_jiffies([run.child.pid]), time.time()
                request = urllib.request.Request(f'http://127.0.0.1:{bench.PORT}/v1/chat/completions', data=body, headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = response.read()
                wall = time.time()-t0
                foreign = max(0,(bench.busy_jiffies()-b0-bench.proc_jiffies([run.child.pid])+o0)/bench.HZ/wall)
                run.event(kind='quality_request', wall_s=wall, foreign_cores=foreign,
                          request_sha256=hashlib.sha256(body).hexdigest(), response_sha256=hashlib.sha256(payload).hexdigest())
                with (out/'responses.jsonl').open('a') as f:
                    f.write(json.dumps(dict(request=json.loads(body), response=json.loads(payload)))+'\n')
                self.send_response(200)
            except Exception as error:
                payload = json.dumps(dict(error=str(error))).encode()
                self.send_response(500)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    try:
        run.quiet()
        run.start(name,cfg)
        run.request(name,cfg,'Say hello in five words.',16,'warmup')
        proxy=ThreadingHTTPServer(('127.0.0.1',18796),Proxy)
        threading.Thread(target=proxy.serve_forever,daemon=True).start()
        with (out/'panel.txt').open('w') as f:
            child=subprocess.Popen(['node',str(HERE/'panel.mjs'),str(out)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                run.renew()
                if time.time()>run.end:
                    raise TimeoutError('quality deadline')
                time.sleep(2)
        if child.returncode:
            raise RuntimeError('panel failed to run')
        run.event(kind='complete', verdict=json.loads((out/'panel-verdict.json').read_text()))
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if proxy:
            proxy.shutdown()
        run.stop()
        run.event(kind='released')


if __name__ == '__main__':
    main()
