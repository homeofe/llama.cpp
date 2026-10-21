#!/usr/bin/env python3
"""Run the arithmetic oracle under the gateway GPU hold; also used for deliberate cuts."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from run_variants import GraphRun, bench, HERE

ap = argparse.ArgumentParser()
ap.add_argument('--runtime', type=Path, required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--expect-failure', action='store_true')
ap.add_argument('--capacities', nargs='+', type=int, default=[1, 2, 4, 8])
args = ap.parse_args()
out = HERE / args.label
out.mkdir(exist_ok=False)
run = GraphRun(out, time.time() + 120)
results = []
try:
    run.held = True
    run.renew(force=True)
    bench.gw('POST', '/api/v1/gpu/free', {})
    if any(mib > 300 for mib in run.gpu_apps().values()):
        raise RuntimeError('GPU already occupied')
    for cap in args.capacities:
        env = dict(os.environ, LD_LIBRARY_PATH=str(args.runtime.resolve()),
                   GGML_CUDA_GRAPH_VARIANTS=str(cap), GGML_CUDA_GRAPH_TRACE='1')
        result = subprocess.run([str(HERE / 'check_graph_variants')], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        (out / f'cap{cap}.txt').write_bytes(result.stdout)
        log = result.stdout.decode(errors='replace')
        counts = dict(collections.Counter(re.findall(r'graph_variant action=(\w+)', log)))
        results.append(dict(capacity=cap, exit_code=result.returncode, actions=counts,
                            oracle=re.findall(r'checks=\d+ failures=\d+', log),
                            runtime_cuda_sha256=hashlib.sha256((args.runtime / 'libggml-cuda.so.0.27.0').read_bytes()).hexdigest()))
        if args.expect_failure:
            assert result.returncode == 1 and 'failures=0' not in log, results[-1]
        else:
            assert result.returncode == 0 and 'checks=264 failures=0' in log, results[-1]
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))
finally:
    run.stop()
