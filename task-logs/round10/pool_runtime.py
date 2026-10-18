#!/usr/bin/env python3
"""Placement, persistent-worker reuse and context-before-pool lifetime checks."""
import argparse
import collections
import json
from pathlib import Path
import resource
import signal
import time

import bench


def snapshot(pid):
    result = {}
    for path in Path(f'/proc/{pid}/task').glob('*/status'):
        try:
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            result[path.parent.name] = fields['Cpus_allowed_list'].strip()
        except FileNotFoundError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen', 'ling'])
    parser.add_argument('tag')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    assert args.tag.replace('-', '').isalnum()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    bench.configure()
    root = bench.ROOT
    out = root / 'task-logs/round10' / ('pool-' + args.tag)
    out.mkdir(exist_ok=False)
    cfg = json.loads((root / 'task-logs/round10/pool-configs.json').read_text())['qwen-pool-native']
    if args.model == 'ling':
        ling = bench.stage.CONFIGS['ling-cache-mtp3']
        placement = cfg['cmd'][cfg['cmd'].index('--cpu-mask'):]
        cfg = dict(cmd=ling['cmd'] + placement, env={**ling['env'],
            **{k: v for k, v in cfg['env'].items() if k.startswith('LD_')}})
    cfg['env']['LD_PRELOAD'] = str(root / 'task-logs/round10/pool-guard.so') + ':' + cfg['env']['LD_PRELOAD']
    if args.baseline:
        baseline = root / 'task-logs/round10/pool-baseline-bin'
        cfg['cmd'][0] = str(baseline / 'llama-server')
        cfg['env']['LD_LIBRARY_PATH'] = str(baseline)
    (out / 'command.json').write_text(json.dumps(cfg, indent=2) + '\n')
    run = bench.Run(out, min(time.time() + 360, 1792320900))
    name = 'pool-check'
    process = None
    try:
        run.quiet()
        run.start(name, cfg)
        process = run.child
        (out / 'owned-pid.json').write_text(json.dumps(dict(pid=process.pid)) + '\n')
        (out / 'maps.txt').write_text(Path(f'/proc/{process.pid}/maps').read_text())
        request_results, snapshots, create_counts = [], [], []
        for index in range(2):
            run.quiet()
            request_results.append(run.request(name, cfg, 'Count from one to five.', 16))
            snapshots.append(snapshot(process.pid))
            create_counts.append((out / (name + '.server.log')).read_text(errors='replace').count('R10_POOL create '))
        (out / 'requests.json').write_text(json.dumps(request_results, indent=2) + '\n')
        (out / 'affinity.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    finally:
        run.stop()
    log = (out / (name + '.server.log')).read_text(errors='replace')
    events = [line[line.index('R10_POOL '):] for line in log.splitlines() if 'R10_POOL ' in line]
    (out / 'events.txt').write_text('\n'.join(events) + '\n')
    attaches = [x for x in events if x.startswith('R10_POOL attach ')]
    frees = [x for x in events if x.startswith('R10_POOL free ')]
    counts = [collections.Counter(x.values()) for x in snapshots]
    worker_sets = [{tid for tid, mask in s.items() if mask.isdigit() and int(mask) < 23} for s in snapshots]
    verdict = dict(model=args.model, baseline=args.baseline, create_counts=create_counts,
        attaches=len(attaches), frees=len(frees), returncode=process.returncode,
        affinity_counts=[dict(c) for c in counts], persistent_worker_tids_equal=worker_sets[0] == worker_sets[1],
        placement_ok=all(all(c[str(cpu)] == 2 for cpu in range(23)) for c in counts),
        teardown_ok='R10_POOL violation' not in log and len(frees) == 2 and process.returncode == 0)
    verdict['passed'] = (len(attaches) == 2 and create_counts == [2, 2] and verdict['placement_ok']
                         and verdict['persistent_worker_tids_equal'] and verdict['teardown_ok'])
    (out / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
    print(json.dumps(verdict), flush=True)
    assert verdict['passed'], 'persistent draft pool contract failed'


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
