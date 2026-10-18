#!/usr/bin/env python3
"""Round 11: reuse the full-pass foreign-process gate and archive cache counters."""
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('round10_bench', ROOT / 'task-logs/round10/bench.py')
assert spec and spec.loader
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
stage = previous.stage


class Run(previous.Run):
    def start(self, name, cfg):
        cache = ROOT / 'task-logs/round11/cuda-cache'
        cache.mkdir(exist_ok=True)
        cfg = {**cfg, 'env': {**cfg.get('env', {}), 'CUDA_CACHE_PATH': str(cache)}}
        super().start(name, cfg)
        directory = Path(cfg['cmd'][0]).resolve().parent
        if directory.name in ['baseline-bin', 'candidate-bin']:
            libraries = sorted({line.split()[-1] for line in Path(f'/proc/{self.child.pid}/maps').read_text().splitlines()
                if '/libggml' in line or '/libllama' in line})
            assert libraries and all(Path(p).parent == directory for p in libraries), 'runtime libraries were mixed'
            log = (self.out / (name + '.server.log')).read_text(errors='replace')
            if cfg['env'].get('GGML_MOE_HYBRID_ASYNC_UPLOAD') == '1':
                assert 'async upload enabled: separate stream' in log, 'CUDA upload path did not activate'
                assert 'async upload unavailable' not in log, 'CUDA upload path fell back'
            self.event(kind='runtime_loaded', config=name, pid=self.child.pid, libraries=libraries,
                cuda_cache_path=str(cache))

    def renew(self, force=False):
        if self.held and (force or time.monotonic() - self.last_hold > 300):
            stage.gw('POST', '/api/v1/gpu/hold', {'reason': 'codex: round 11', 'ttlS': 900})
            self.last_hold = time.monotonic()

    def one_pass(self, name, cfg, labels, rnd):
        log = self.out / (name + '.server.log')
        start = log.stat().st_size if log.exists() else 0
        try:
            super().one_pass(name, cfg, labels, rnd)
        finally:
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(start)
                    tail = stream.read().decode(errors='replace')
                self.event(kind='cache_pass', config=name, round=rnd,
                    completed=all(label in self.done for label in labels),
                    speculative_rounds=len(re.findall(r'generate_draft:.*#draft=\d+', tail)),
                    cache_statistics=re.findall(r'llama_moe_cache: (?:context = .*statistics follow|adaptive:.*)', tail),
                    prefill_bursts=re.findall(r'llama_moe_cache: prefill burst:.*', tail))


def configure():
    previous.configure()
    stage.CONFIGS.update(json.loads(Path(os.environ.get('R11_CONFIGS', str(Path(__file__).with_name('configs.json')))).read_text()))
    stage.Run = Run


if __name__ == '__main__':
    configure()
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'configs.json').write_text(json.dumps(stage.CONFIGS, indent=2) + '\n')
    stage.main()
