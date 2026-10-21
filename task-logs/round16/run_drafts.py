#!/usr/bin/env python3
"""Interleave external-MTP candidates with bounded quiet waits and full outputs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'round15'))
from probe import Probe, bench

MODELS = Path('/opt/llm-gateway/data/moe-quant/qwen38/round16-models')


class Contaminated(Exception):
    pass


class Run(Probe):
    def renew(self, force=False):
        if self.held and (force or time.monotonic() - self.last_hold > 300):
            bench.gw('POST', '/api/v1/gpu/hold', {'reason': 'codex: round16 Strata draft parity', 'ttlS': 900})
            self.last_hold = time.monotonic()

    def start(self, name, cfg):
        self.held = True
        self.renew(force=True)
        bench.gw('POST', '/api/v1/gpu/free', {})
        if any(mib > 300 for mib in self.gpu_apps().values()):
            raise RuntimeError('GPU already in use')
        env = {k:v for k,v in os.environ.items() if not k.startswith('GGML_MOE_HYBRID') and k != 'LD_LIBRARY_PATH'}
        env.update(cfg['env'])
        runtime = Path(cfg['cmd'][0]).resolve().parent
        env['LD_LIBRARY_PATH'] = str(runtime)
        with (self.out / (name + '.server.txt')).open('a') as log:
            self.child = subprocess.Popen(cfg['cmd'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = min(self.end, time.time() + 180)
        while time.time() < deadline:
            if self.child.poll() is not None:
                raise RuntimeError(f'server exited {self.child.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{bench.PORT}/health', timeout=2) as f:
                    if f.status == 200:
                        paths = sorted({line.split()[-1] for line in Path(f'/proc/{self.child.pid}/maps').read_text().splitlines()
                                        if '/' in line and any(lib in line for lib in ('libggml', 'libllama', 'libmtmd'))})
                        self.event(kind='loaded_libraries', config=name, runtime=str(runtime), paths=paths)
                        if not paths or any(Path(path).parent != runtime for path in paths):
                            raise RuntimeError('runtime loaded libraries outside its isolated directory')
                        return
            except (OSError, ValueError):
                pass
            self.renew()
            time.sleep(2)
        raise TimeoutError('server startup deadline')


def configs():
    base = bench.CONFIGS['q-vibe-async']
    result = {}
    for name in ('base', 'base-new', 'q2', 'q2id', 'q8sub', 'q6sub', 'resident', 'resident-n4', 'q8gpu', 'q6gpu', 'q6gpu-cache', 'q6gpu-n4', 'q8exp-q8sub', 'q8exp-q8sub-cache', 'q8exp-q6sub', 'q8exp-q6sub-cache', 'q6gpu-p7', 'q6gpu-global', 'strata-gpu', 'strata-gpu-global'):
        cfg = dict(cmd=list(base['cmd']), env=dict(base['env']))
        cfg['env'].pop('LD_LIBRARY_PATH', None)
        if name == 'base':
            cfg['cmd'][0] = str(HERE / 'e2e/original/llama-server')
        else:
            cfg['cmd'][0] = str(HERE / 'e2e/isolated/llama-server')
        if name not in ('base', 'base-new'):
            variant = {'q2':'q2', 'q2id':'q2-identity', 'q8sub':'q2-q8subset', 'q8gpu':'q2-q8subset'}.get(name, 'q2-q6subset')
            if name.startswith('q8exp-'):
                variant = 'q8exp-q8subset' if name.startswith('q8exp-q8sub') else 'q8exp-q6subset'
            if name.startswith('strata-'):
                variant = 'strata-q6subset'
            cfg['cmd'][cfg['cmd'].index('-md')+1] = str(MODELS / (variant + '.gguf'))
        if name.startswith('resident'):
            cfg['env'].update(GGML_MOE_HYBRID_DRAFT_MIB='704', GGML_MOE_HYBRID_DRAFT_STATIC_PCT='100',
                              GGML_MOE_HYBRID_DRAFT_PROFILE=str(MODELS / 'all-experts.profile'))
        if 'gpu' in name:
            cfg['cmd'].remove('-cmoed')
            cfg['cmd'] += ['--spec-draft-ngl', '99']
            cfg['env']['GGML_MOE_HYBRID_DRAFT_MIB'] = '0'
        if name == 'q6gpu-cache':
            cfg['env']['GGML_MOE_HYBRID_MIB'] = '3896'
        if name == 'q8exp-q6sub-cache':
            cfg['env']['GGML_MOE_HYBRID_MIB'] = '4024'
        if name == 'q8exp-q8sub-cache':
            cfg['env']['GGML_MOE_HYBRID_MIB'] = '3960'
        if name == 'q6gpu-p7':
            cfg['cmd'] += ['--spec-draft-p-min', '0.7']
        if name in ('q6gpu-global', 'strata-gpu-global'):
            cfg['cmd'][0] = str(HERE / 'e2e/global/llama-server')
            cfg['env']['LLAMA_MTP_HNORM_GLOBAL'] = '1'
        if name in ('resident-n4', 'q6gpu-n4'):
            cfg['cmd'][cfg['cmd'].index('--spec-draft-n-max')+1] = '4'
        result[name] = cfg
    cfg = dict(cmd=list(result['strata-gpu-global']['cmd']), env=dict(result['strata-gpu-global']['env']))
    cfg['cmd'][0] = str(HERE / 'e2e/reference-packed-v2/llama-server')
    cfg['cmd'][cfg['cmd'].index('--spec-draft-n-max')+1] = '5'
    cfg['cmd'] += ['--spec-draft-p-min', '0.7']
    cfg['env'].update(LLAMA_MTP_STRATA_REFERENCE='1', LLAMA_MTP_DENSE='1', GGML_MOE_HYBRID_ADAPTIVE='0',
                      GGML_MOE_HYBRID_MIB='3600', GGML_MOE_HYBRID_PACK_PROFILE='1',
                      GGML_MOE_HYBRID_PROFILE=str(HERE/'strata-target.profile'))
    result['strata-reference'] = cfg
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=['base','q2','q8sub','q6sub','resident'])
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--all-prompts', action='store_true')
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--minutes', type=int, default=30)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(exist_ok=False)
    cfgs = configs()
    (out/'configs.json').write_text(json.dumps({n:cfgs[n] for n in a.configs}, indent=2)+'\n')
    run = Run(out, time.time()+a.minutes*60)
    def stop(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    prompts = bench.PROMPTS if a.all_prompts else [bench.PROMPTS[0], bench.PROMPTS[4]]
    try:
        for rnd in range(a.rounds):
            order = a.configs if rnd % 2 == 0 else a.configs[::-1]
            for name in order:
                for block_attempt in range(3):
                    try:
                        run.quiet()
                        run.event(kind='start_config', config=name, round=rnd, block_attempt=block_attempt)
                        run.start(name, cfgs[name])
                        prefix = f'{name}-r{rnd}-b{block_attempt}'
                        run.request(name, cfgs[name], 'Say hello in five words.', 16, prefix+'-warmup')
                        for prompt_id, prompt in prompts:
                            waited, _ = run.quiet()
                            b0, o0 = bench.busy_jiffies(), bench.proc_jiffies([run.child.pid])
                            load = os.getloadavg()[0]
                            tag = prefix+'-'+prompt_id
                            res = run.request(name, cfgs[name], prompt, a.tokens, tag)
                            foreign = max(0, (bench.busy_jiffies()-b0-bench.proc_jiffies([run.child.pid])+o0)/bench.HZ/res['wall_s'])
                            run.event(kind='request', config=name, round=rnd, prompt=prompt_id, block_attempt=block_attempt,
                                      load_start=load, load_end=os.getloadavg()[0], waited_s=waited,
                                      foreign_cores=foreign, accepted=load < 2 and foreign < 1,
                                      gpu=run.gpu_apps(), **res)
                            if foreign >= 1 or load >= 2:
                                raise Contaminated('foreign load; reset cache and replay the entire block')
                        run.event(kind='block_complete', config=name, round=rnd, block_attempt=block_attempt)
                        break
                    except Contaminated as error:
                        run.event(kind='block_discarded', config=name, round=rnd, block_attempt=block_attempt, error=str(error))
                        if block_attempt == 2:
                            raise
                    except Exception as error:
                        run.event(kind='error', config=name, round=rnd, error=str(error))
                        raise
                    finally:
                        run.stop()
        run.event(kind='complete')
    finally:
        run.stop()
        run.event(kind='released')


if __name__ == '__main__':
    main()
