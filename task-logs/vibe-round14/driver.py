#!/usr/bin/env python3
"""Round 14 driver: runs INSIDE nsys. Starts llama-server (q-vibe-async, graphs off),
health + warmup, then pushes the NVTX range 'round14' around the quiet-gated steady
code request. --capture-range-end=stop in the runner makes nsys finalize the report the
moment the range pops, independent of the server's slow 83 GB shutdown."""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/opt/llm-gateway/data/build/llama-vibe/task-logs/vibe-round1')
sys.path.insert(0, '/opt/llm-gateway/data/moe-quant/qwen38')
import stage1  # noqa: E402
import stage_b  # noqa: E402

ROOT = Path('/opt/llm-gateway/data/build/llama-vibe')
OUT = ROOT / 'task-logs/vibe-round14'


def main():
    prompt_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = time.time() + 3600
    cfg = stage_b.CONFIGS['q-vibe-async']
    env = dict(os.environ, **cfg['env'], GGML_CUDA_DISABLE_GRAPHS='1')
    ours = str(ROOT / 'build-cuda/bin/llama-server')
    cmd = [(ours if c == stage_b.L5 else c) for c in cfg['cmd']]
    port = int(cmd[cmd.index('--port') + 1])
    url = f'http://127.0.0.1:{port}'

    log = stage1.Ledger(OUT)
    log(kind='driver_start', prompt=stage_b.PROMPTS[prompt_idx][0])

    server_log = (OUT / 'driver-server.log').open('w')
    child = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    (OUT / 'driver-pids.json').write_text(json.dumps(dict(driver=os.getpid(), server=child.pid)))

    nvtx = ctypes.CDLL('libnvToolsExt.so.1')
    nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    nvtx.nvtxRangePushA.restype = ctypes.c_int
    nvtx.nvtxRangePop.restype = ctypes.c_int
    nvtx.nvtxMarkA.argtypes = [ctypes.c_char_p]

    try:
        stage1.wait_ready(url, '', child, end, log)
        warm = stage1.request(url, '', 'Say hello in five words.', 16)
        log(kind='warmup', tokens=warm['tokens'])

        nvtx.nvtxRangePushA(b'round14')
        nvtx.nvtxMarkA(('epoch_ns=' + str(time.time_ns())).encode())
        res = stage1.gate_request(url, '', stage_b.PROMPTS[prompt_idx][1],
                                  'steady', log, end, sampler_pid=child.pid)
        nvtx.nvtxRangePop()
        log(kind='request_done', prompt=stage_b.PROMPTS[prompt_idx][0], **res)
    finally:
        child.send_signal(signal.SIGINT)
        deadline = time.time() + 300
        while child.poll() is None and time.time() < deadline:
            time.sleep(2)
        if child.poll() is None:
            child.kill()
        server_log.close()
        log(kind='server_stopped', rc=child.poll())


if __name__ == '__main__':
    main()
