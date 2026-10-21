#!/usr/bin/env python3
"""Fresh-process, paired-output A/B benchmark for bounded CUDA graph variants."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'round16'))
from native_reference import Reference, configs, bench
from run_drafts import Contaminated


class GraphRun(Reference):
    def renew(self, force=False):
        if self.held and (force or time.monotonic() - self.last_hold > 300):
            bench.gw('POST', '/api/v1/gpu/hold', {'reason': 'codex: bounded CUDA graph reuse measurement', 'ttlS': 900})
            self.last_hold = time.monotonic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--stack', choices=['aligned', 'best'], default='aligned')
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--rounds', type=int, choices=[1, 3], default=1)
    ap.add_argument('--capacity', type=int, default=4)
    ap.add_argument('--minutes', type=int, default=20)
    args = ap.parse_args()
    out = HERE / args.out
    out.mkdir(exist_ok=False)
    cfg = configs()['strata-reference' if args.stack == 'aligned' else 'base']
    controls = {'A': copy.deepcopy(cfg), 'B': copy.deepcopy(cfg)}
    controls['B']['cmd'][0] = str(args.candidate.resolve() / 'llama-server')
    controls['B']['env']['GGML_CUDA_GRAPH_VARIANTS'] = str(args.capacity)
    for control in controls.values():
        control['env']['GGML_CUDA_GRAPH_TRACE'] = '0'
    (out / 'configs.json').write_text(json.dumps(controls, indent=2) + '\n')
    run = GraphRun(out, time.time() + args.minutes * 60)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        for block, name in enumerate(['A', 'B'] if args.rounds == 1 else ['A', 'B', 'B', 'A', 'A', 'B']):
            for attempt in range(3):
                try:
                    run.quiet()
                    run.event(kind='start_config', config=name, block=block, attempt=attempt, stack=args.stack)
                    run.start(name, controls[name])
                    prefix = f'{block}-{name}-{attempt}'
                    run.request('Say hello in five words.', prefix + '-warmup', 16)
                    for prompt_id, prompt in [bench.PROMPTS[0], bench.PROMPTS[4]]:
                        for phase in ('novel', 'repeat'):
                            waited, _ = run.quiet()
                            load = os.getloadavg()[0]
                            b0, o0 = bench.busy_jiffies(), bench.proc_jiffies(run.owned())
                            result = run.request(prompt, prefix + '-' + prompt_id + '-' + phase)
                            foreign = max(0, (bench.busy_jiffies() - b0 - bench.proc_jiffies(run.owned()) + o0) / bench.HZ / result['wall_s'])
                            run.event(kind='request', config=name, block=block, attempt=attempt, stack=args.stack,
                                      prompt=prompt_id, phase=phase, load_start=load, load_end=os.getloadavg()[0],
                                      waited_s=waited, foreign_cores=foreign, accepted=load < 2 and foreign < 1,
                                      gpu=run.gpu_apps(), **result)
                            if load >= 2 or foreign >= 1:
                                raise Contaminated('foreign load; discard entire process and repeat')
                    run.event(kind='block_complete', config=name, block=block, attempt=attempt)
                    break
                except Contaminated as error:
                    run.event(kind='block_discarded', config=name, block=block, attempt=attempt, error=str(error))
                    if attempt == 2:
                        raise
                finally:
                    run.stop()
        run.event(kind='complete')
    finally:
        run.stop()
        run.event(kind='released', gpu=run.gpu_apps())


if __name__ == '__main__':
    main()
