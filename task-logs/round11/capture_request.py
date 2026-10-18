#!/usr/bin/env python3
"""One NVTX-bounded request for copy/compute overlap; the parent owns the GPU lease."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('config')
    args = parser.parse_args()
    injection = os.environ.get('LD_PRELOAD')
    bench.configure()
    if injection:
        os.environ['LD_PRELOAD'] = injection
    out = args.out.resolve()
    cfg = json.loads(json.dumps(bench.stage.CONFIGS[args.config]))
    cfg['env'].update(GGML_CUDA_DISABLE_GRAPHS='1', CUDA_CACHE_PATH=str(out / 'cuda-cache'))
    cfg['cmd'] += ['-lv', '4']
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('profile10', bench.ROOT / 'task-logs/round10/profile_request.py')
    assert spec and spec.loader
    profile10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile10)
    nvtx = ctypes.CDLL('libnvToolsExt.so.1')
    nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    nvtx.nvtxMarkA.argtypes = [ctypes.c_char_p]
    run = bench.Run(out, min(time.time() + 1200, 1792335540))
    try:
        with (out / (args.config + '.server.log')).open('w') as stream:
            run.child = subprocess.Popen(cfg['cmd'], env={**os.environ, **cfg['env']}, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        run.gpu_pid = run.child.pid
        (out / 'owned-pids.json').write_text(json.dumps(dict(wrapper=os.getpid(), server=run.child.pid)) + '\n')
        for _ in range(450):
            if run.child.poll() is not None:
                raise RuntimeError('server exited before health')
            try:
                with bench.stage.urllib.request.urlopen('http://127.0.0.1:8799/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise TimeoutError('server health')
        run.request(args.config, cfg, 'Say hello in five words.', 16)
        run.quiet()
        run.check_gpu()
        root_pid = json.loads((out / 'profiler-owner.json').read_text())['pid']
        pids = profile10.owned_processes(root_pid, out)
        before_busy, before_own = bench.stage.busy_jiffies(), bench.stage.proc_jiffies(pids)
        start = time.monotonic()
        original = bench.stage.urllib.request.urlopen
        first = False

        class Response:
            def __init__(self, response):
                self.response = response

            def __enter__(self):
                self.response.__enter__()
                return self

            def __exit__(self, *exc):
                return self.response.__exit__(*exc)

            def __iter__(self):
                nonlocal first
                for raw in self.response:
                    if not first and raw.startswith(b'data:') and b'[DONE]' not in raw:
                        data = json.loads(raw[5:])
                        if any((c.get('delta') or {}).get('content') or (c.get('delta') or {}).get('reasoning_content') for c in data.get('choices', [])):
                            first = True
                            nvtx.nvtxMarkA(b'first_token')
                    yield raw

        nvtx.nvtxRangePushA(b'round11')
        nvtx.nvtxRangePushA(b'code-duration')
        bench.stage.urllib.request.urlopen = lambda *a, **kw: Response(original(*a, **kw))
        try:
            result = run.request(args.config, cfg, dict(bench.stage.PROMPTS)['code-duration'], 256)
        finally:
            bench.stage.urllib.request.urlopen = original
            nvtx.nvtxRangePop()
            nvtx.nvtxRangePop()
        wall = time.monotonic() - start
        foreign = max(0, bench.stage.busy_jiffies() - before_busy - bench.stage.proc_jiffies(pids) + before_own) / bench.stage.HZ / wall
        run.event(kind='capture_request', config=args.config, foreign_cores=foreign, accepted=foreign < 1,
            owned_profiler_processes=pids, wall_s=wall, **result)
        assert foreign < 1, 'profile foreign CPU gate failed'
    finally:
        run.stop()
        (out / 'server-stopped.json').write_text(json.dumps(dict(epoch=time.time())) + '\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
