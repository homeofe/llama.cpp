#!/usr/bin/env python3
"""Prepare a bounded CPU screen; execute only after the multiplicity decision."""
import argparse
import copy
import json
from pathlib import Path

import bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='q-warm-draft')
    args = parser.parse_args()
    path = Path(__file__).with_name('extra-configs.json')
    extra = json.loads(path.read_text())
    configs = {**bench.stage.CONFIGS, **extra}
    base = configs[args.base]
    variants = [
        ('qwen-cpu-control', None, None, 300000),
        ('qwen-cpu-spin1000', None, None, 1000),
        ('qwen-cpu-t16-spread', 16, '6db6db', 300000),
        ('qwen-cpu-t24-physical', 24, 'ffffff', 300000),
        ('qwen-cpu-t48-smt', 48, 'ffffffffffff', 300000),
    ]
    for name, threads, mask, spin in variants:
        cfg = copy.deepcopy(base)
        cfg['env']['GOMP_SPINCOUNT'] = str(spin)
        if threads is not None:
            cfg['cmd'][cfg['cmd'].index('-t') + 1] = str(threads)
            cfg['cmd'][cfg['cmd'].index('-tb') + 1] = str(threads)
            # Explicit draft settings avoid silently inheriting target masks.
            # The experiment changes the complete decode placement for both contexts.
            cfg['cmd'] += ['--cpu-mask', mask, '--cpu-strict', '1',
                           '--cpu-mask-batch', mask, '--cpu-strict-batch', '1',
                           '--spec-draft-threads', str(threads), '--cpu-mask-draft', mask,
                           '--cpu-strict-draft', '1', '--spec-draft-threads-batch', str(threads),
                           '--cpu-mask-batch-draft', mask, '--cpu-strict-batch-draft', '1']
        extra[name] = cfg
    path.write_text(json.dumps(extra, indent=2) + '\n')
    manifest = dict(base=args.base, variants=[row[0] for row in variants],
                    changes='Target and draft threads/masks, including batch threads used for four-token verification; OMP_PLACES=cores.',
                    costs='About 15 minutes for one screening round; repeat only a promising winner.',
                    threshold='At least 3% on matching per-prompt work before recommending a new default.')
    path.with_name('cpu-sweep-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(' '.join(manifest['variants']))


if __name__ == '__main__':
    main()
