#!/usr/bin/env python3
"""Real-model integration only: shared CPU load is allowed; rates are not benchmark results."""
import argparse
import json
import os
import signal
import time

from native_reference import Reference
from run_drafts import HERE, bench, configs

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--qsa', action='store_true')
a = ap.parse_args()
out = HERE/a.out
out.mkdir(exist_ok=False)
cfg = configs()['strata-reference']
cfg['cmd'] += ['-lv', '4']
if a.qsa:
    cfg['env']['LLAMA_MTP_DENSE'] = '0'
(out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
run = Reference(out, time.time()+300)
def stop(*_):
    raise KeyboardInterrupt('SIGTERM')
signal.signal(signal.SIGTERM, stop)
try:
    run.event(kind='correctness_only', benchmark_valid=False, load=os.getloadavg())
    run.start('strata-reference', cfg)
    for prompt_id, prompt in [bench.PROMPTS[0], bench.PROMPTS[3], bench.PROMPTS[4]]:
        result = run.request(prompt, prompt_id)
        assert 0 < result['tokens'] <= 256, 'invalid completion length'
        run.event(kind='integration_request', prompt=prompt_id, benchmark_valid=False,
                  load=os.getloadavg(), gpu=run.gpu_apps(), **result)
    run.event(kind='complete')
finally:
    run.stop()
    run.event(kind='released', gpu=run.gpu_apps())
