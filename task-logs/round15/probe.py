#!/usr/bin/env python3
"""Bound the n5 path with the existing stage_b model, prompts and server lifecycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, '/opt/llm-gateway/data/moe-quant/qwen38')
import stage_b as bench

ROOT = Path(__file__).resolve().parent


class Probe(bench.Run):
    def quiet(self):
        start = time.time()
        while time.time() < self.end:
            self.renew()
            b0 = bench.busy_jiffies()
            o0 = bench.proc_jiffies([self.child.pid]) if self.child else 0
            time.sleep(2)
            own = bench.proc_jiffies([self.child.pid]) - o0 if self.child else 0
            foreign = max(0, (bench.busy_jiffies() - b0 - own) / bench.HZ / 2)
            if os.getloadavg()[0] < 2 and foreign < 1:
                return time.time() - start, foreign
        raise TimeoutError('quiet-window deadline')

    def start(self, name, cfg):
        self.held = True
        self.renew(force=True)
        super().start(name, cfg)

    def request(self, name, cfg, prompt, limit, tag):
        body = dict(messages=[dict(role='user', content=prompt)], max_tokens=limit,
                    temperature=0, seed=1, stream=True, stream_options=dict(include_usage=True),
                    chat_template_kwargs=dict(enable_thinking=False), cache_prompt=False)
        req = urllib.request.Request(f'http://127.0.0.1:{bench.PORT}/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        first = last = None
        chunks = []
        usage = timings = None
        finished = False
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    finished = True
                    break
                event = json.loads(payload)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                usage = event.get('usage') or usage
                timings = event.get('timings') or timings
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    text = delta.get('content') or delta.get('reasoning_content')
                    if text:
                        last = time.monotonic()
                        first = first or last
                        chunks.append(text)
        assert finished and usage and chunks, 'incomplete generation'
        text = ''.join(chunks)
        (self.out / f'{tag}.txt').write_text(text)
        tokens = usage['completion_tokens']
        return dict(tokens=tokens, sha256=hashlib.sha256(text.encode()).hexdigest(),
                    decode_tok_s=(tokens - 1) / (last - first), ttft_s=first-start,
                    wall_s=time.monotonic()-start, timings=timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=['n3', 'n5-reserve'])
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--minutes', type=float, default=15)
    parser.add_argument('--output', default='screen')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    baseline = bench.CONFIGS['q-vibe-async']
    configs = {}
    for name in ('n3', 'n5', 'n5-reserve', 'n5-nograph'):
        cfg = dict(cmd=list(baseline['cmd']), env=dict(baseline['env']))
        cfg['env'].pop('LD_LIBRARY_PATH', None)
        if name != 'n3':
            cfg['cmd'][cfg['cmd'].index('--spec-draft-n-max') + 1] = '5'
        if name == 'n5-reserve':
            cfg['env']['GGML_MOE_HYBRID_MIB'] = '3384'
        if name == 'n5-nograph':
            cfg['env']['GGML_CUDA_DISABLE_GRAPHS'] = '1'
        configs[name] = cfg
    (out / 'configs.json').write_text(json.dumps(configs, indent=2) + '\n')
    run = Probe(out, time.time() + args.minutes * 60)
    def stop(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    prompts = [bench.PROMPTS[0], bench.PROMPTS[4]]
    try:
        for rnd in range(args.rounds):
            order = args.configs if rnd % 2 == 0 else list(reversed(args.configs))
            for name in order:
                try:
                    run.quiet()
                    run.event(kind='start_config', config=name, round=rnd)
                    cfg = configs[name]
                    run.start(name, cfg)
                    run.event(kind='loaded', config=name, gpu=run.gpu_apps())
                    run.request(name, cfg, 'Say hello in five words.', 16, f'{name}-r{rnd}-warmup')
                    for prompt_id, prompt in prompts:
                        waited, pre = run.quiet()
                        tag = f'{name}-r{rnd}-{prompt_id}'
                        b0, o0 = bench.busy_jiffies(), bench.proc_jiffies([run.child.pid])
                        load_start = os.getloadavg()[0]
                        res = run.request(name, cfg, prompt, 256, tag)
                        foreign = max(0, (bench.busy_jiffies()-b0 - (bench.proc_jiffies([run.child.pid])-o0)) / bench.HZ / res['wall_s'])
                        run.event(kind='request', config=name, round=rnd, prompt=prompt_id,
                                  waited_s=waited, load_start=load_start, load_end=os.getloadavg()[0],
                                  foreign_cores=foreign, accepted=foreign < 1,
                                  gpu=run.gpu_apps(), **res)
                except Exception as error:
                    run.event(kind='error', config=name, error=str(error))
                    if isinstance(error, TimeoutError):
                        raise
                finally:
                    run.stop()
        run.event(kind='complete')
    finally:
        run.stop()
        run.event(kind='released')


if __name__ == '__main__':
    main()
