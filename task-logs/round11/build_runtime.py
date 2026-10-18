#!/usr/bin/env python3
"""Isolate the experiment to libllama and the CUDA backend; retain original bins."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import bench

ROOT = bench.ROOT
OUT = ROOT / 'task-logs/round11'


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file() and not p.is_symlink()}


def main():
    assert json.loads((OUT / 'host-complete.json').read_text())['passed']
    assert not json.loads(bench.stage.gw('GET', '/api/v1/status')).get('gpuHold'), 'do not build with a GPU lease'
    baseline = OUT / 'baseline-bin'
    candidate = OUT / 'candidate-bin'
    if not baseline.exists():
        shutil.copytree(ROOT / 'build-cuda/bin', baseline, symlinks=True)
    (OUT / 'baseline-sha256.json').write_text(json.dumps(hashes(baseline), indent=2) + '\n')
    with (OUT / 'cuda-build.log').open('w') as stream:
        subprocess.run(['nice', '-n', '10', 'cmake', '--build', 'build-cuda', '--target', 'llama-server', 'llama-perplexity', '-j', '16'], cwd=ROOT,
            env={**os.environ, 'TMPDIR': str(OUT / 'tmp'), 'CUDA_CACHE_PATH': str(OUT / 'cuda-cache')}, stdout=stream, stderr=subprocess.STDOUT, check=True)
    if not candidate.exists():
        shutil.copytree(baseline, candidate, symlinks=True)
    replaced = ['libllama.so.0.7.0', 'libggml-cuda.so.0.27.0']
    for name in replaced:
        shutil.copy2(ROOT / 'build-cuda/bin' / name, candidate / name)
    before, after = hashes(baseline), hashes(candidate)
    assert sorted(k for k in before if before[k] != after[k]) == sorted(replaced)
    (OUT / 'runtime-isolation.json').write_text(json.dumps(dict(replaced=replaced, source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        baseline_sha256=before, candidate_sha256=after), indent=2) + '\n')
    bench.configure()
    configs = json.loads((OUT / 'configs.json').read_text())
    for model, original in [('qwen', 'q-warm-draft'), ('ling', 'ling-dspark-n3-p00')]:
        for variant, directory in [('control', baseline), ('async', candidate)]:
            cfg = json.loads(json.dumps(bench.stage.CONFIGS[original]))
            cfg['cmd'][0] = str(directory / 'llama-server')
            cfg['env']['LD_LIBRARY_PATH'] = str(directory)
            cfg['env']['GGML_MOE_HYBRID_ASYNC_UPLOAD'] = '1' if variant == 'async' else '0'
            configs[model + '-' + variant] = cfg
    (OUT / 'configs.json').write_text(json.dumps(configs, indent=2) + '\n')


if __name__ == '__main__':
    main()
