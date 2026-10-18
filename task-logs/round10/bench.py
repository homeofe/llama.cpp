#!/usr/bin/env python3
"""Stage B gates with full-pass retries, output hashes and per-request draft lengths."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('round9_bench', ROOT / 'task-logs/round9/bench.py')
assert spec and spec.loader
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
stage = previous.stage


class Run(previous.Run):
    def renew(self, force=False):
        if self.held and (force or time.monotonic() - self.last_hold > 300):
            stage.gw('POST', '/api/v1/gpu/hold', {'reason': 'codex: round 10', 'ttlS': 900})
            self.last_hold = time.monotonic()

    def request(self, name, cfg, text, max_tokens):
        log = self.out / (name + '.server.log')
        start = log.stat().st_size if log.exists() else 0
        result = super().request(name, cfg, text, max_tokens)
        if log.exists():
            with log.open() as stream:
                stream.seek(start)
                tail = stream.read()
            rows = re.findall(r'draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)', tail)
            if rows:
                accepted, drafted, length = rows[-1]
                timings = result.get('server_timings') or {}
                assert int(accepted) == timings.get('draft_n_accepted')
                assert int(drafted) == timings.get('draft_n')
                result['mean_accepted_length_including_bonus'] = float(length)
        return result

    def one_pass(self, name, cfg, labels, rnd):
        pending = []
        try:
            self.start(name, cfg)
            self.previous_gpu_pids = set(self.gpu_apps()) - {self.child.pid}
            self.gpu_pid = self.child.pid
            self.check_gpu()
            self.request(name, cfg, 'Say hello in five words.', 16)
            for (pid, prompt), label in zip(stage.PROMPTS, labels):
                waited, pre = self.quiet()
                pids = stage.engine_pids(name)
                b0, o0, t0 = stage.busy_jiffies(), stage.proc_jiffies(pids), time.time()
                result = self.request(name, cfg, prompt, 256)
                wall = time.time() - t0
                foreign = max(0.0, (stage.busy_jiffies() - b0 - stage.proc_jiffies(pids) + o0) / stage.HZ / wall)
                self.check_gpu()
                self.renew()
                record = dict(kind='request', label=label, config=name, round=rnd, prompt=pid,
                              waited_s=waited, pre_busy_cores=pre, gate='foreign', wall_s=wall,
                              foreign_cores=foreign, vram_mib=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip(),
                              accepted=foreign < 1.0 and result['decode_tok_s'] is not None, **result)
                self.event(**{**record, 'kind': 'request_pending'})
                if not record['accepted']:
                    raise stage.Requeue('foreign CPU gate failed; restart entire pass with a fresh cache')
                pending.append(record)
            for record in pending:
                self.event(**record)
                self.done.add(record['label'])
        finally:
            self.stop()


def configure():
    for key in list(os.environ):
        if key.startswith(('GGML_', 'LLAMA_ARG_SPEC_')) or key in ('LD_PRELOAD', 'GOMP_SPINCOUNT'):
            del os.environ[key]
    stage.CONFIGS.update(json.loads((Path(__file__).with_name('configs.json')).read_text()))
    stage.Run = Run


if __name__ == '__main__':
    configure()
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'harness.py').write_bytes(previous.SOURCE.read_bytes())
    (out / 'configs.json').write_text(json.dumps(stage.CONFIGS, indent=2) + '\n')
    stage.main()
