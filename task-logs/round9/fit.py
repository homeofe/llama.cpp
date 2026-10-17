#!/usr/bin/env python3
"""Measure the actual allocation of one draft-ubatch/cache-budget pair."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen', 'ling'])
    parser.add_argument('budget', type=int)
    args = parser.parse_args()
    for key in list(os.environ):
        if key.startswith('GGML_MOE_HYBRID') or key == 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH':
            del os.environ[key]
    control = 'q-warm-draft' if args.model == 'qwen' else 'ling-cache-mtp3'
    base = bench.stage.CONFIGS[control]
    cfg = dict(cmd=base['cmd'] + ['--spec-draft-n-ubatch', '32'],
               env={**base['env'], 'GGML_MOE_HYBRID_MIB': str(args.budget)})
    name = f'{args.model}-fit-{args.budget}'
    out = Path(__file__).resolve().parent / name
    out.mkdir(exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    run = bench.Run(out, min(time.time() + 1200, 1792277760))
    try:
        run.quiet()
        run.start(name, cfg)
        run.check_gpu()
        response = run.request(name, cfg, 'Say hello in five words.', 16)
        run.check_gpu()
        memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                             '--format=csv,noheader,nounits'], text=True).strip())
        run.event(kind='fit', model=args.model, budget_mib=args.budget, vram_mib=memory,
                  fits=memory <= 10700, response=response)
    finally:
        run.stop()


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    main()
