#!/usr/bin/env python3
"""Capture the measured Strata runtime with CUDA graph nodes visible."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'strata-nodes'
sys.path.insert(0, str(ROOT.parent / 'vibe-round1'))
import stage1


def quiet(deadline):
    while time.time() < deadline:
        if os.getloadavg()[0] < 2:
            return
        time.sleep(3)
    raise TimeoutError('quiet window deadline')


def driver():
    log = stage1.Ledger(OUT)
    end = time.time() + 420
    proc = stage1.start_strata(OUT)
    nvtx = ctypes.CDLL('libnvToolsExt.so.1')
    nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    url = f'http://127.0.0.1:{stage1.PORT}'
    try:
        log(kind='server_pid', pid=proc.pid)
        stage1.wait_ready(url, stage1.MYKEY, proc, end, log, timeout=180)
        warm = stage1.request(url, stage1.MYKEY, 'Say hello in five words.', 16)
        log(kind='warmup', **warm)
        quiet(end)
        load = os.getloadavg()[0]
        nvtx.nvtxRangePushA(b'round15-strata')
        try:
            res = stage1.gate_request(url, stage1.MYKEY, stage1.PROMPTS[0][1], 'code-duration', log, end)
        finally:
            nvtx.nvtxRangePop()
        log(kind='request', load_at_capture=load, **res)
        metrics = stage1.http_json(url + '/metrics', key=stage1.MYKEY)
        log(kind='metrics', metrics=stage1.last_history(metrics))
    finally:
        # The engine is a child in this server's process group.
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        log(kind='server_stopped', rc=proc.returncode)
        # The temporary server key has no purpose after shutdown.
        config = OUT / 'config.json'
        if config.exists():
            cfg = json.loads(config.read_text())
            cfg.pop('api_key', None)
            config.write_text(json.dumps(cfg, indent=2) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'trace.nsys-rep').exists() or (OUT / 'trace.qdstrm').exists():
        raise FileExistsError('preserve the previous capture before rerunning')
    log = stage1.Ledger(OUT)
    proc = None
    held = False
    try:
        quiet(time.time() + 180)
        stage1.gw('POST', '/api/v1/gpu/hold', {'reason': 'round15 Strata graph-node trace', 'ttlS': 900})
        held = True
        stage1.gw('POST', '/api/v1/gpu/free', {})
        log(kind='hold_acquired')
        cmd = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
               '--cuda-graph-trace=node', '--capture-range=nvtx', '--nvtx-capture=round15-strata',
               '--capture-range-end=stop', '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0',
               '--kill=none', '--duration=420', '--export=sqlite', '-o', str(OUT / 'trace'),
               sys.executable, str(Path(__file__).resolve()), 'driver']
        log(kind='profile_command', argv=cmd)
        with (OUT / 'nsys.txt').open('w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            proc.wait(timeout=540)
        log(kind='profile_exit', rc=proc.returncode)
    except Exception as error:
        log(kind='error', error=str(error), load=os.getloadavg()[0])
        raise
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if held:
            stage1.gw('DELETE', '/api/v1/gpu/hold')
            log(kind='hold_released')


if __name__ == '__main__':
    if sys.argv[1:] == ['driver']:
        driver()
    else:
        main()
