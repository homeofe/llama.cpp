#!/usr/bin/env python3
"""After the two-engine baseline releases the GPU, validate and run its aligned candidate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_drafts import Run, HERE, bench

pid = 1048159
while Path(f'/proc/{pid}/cmdline').exists():
    if b'task-logs/round16/native_reference.py' not in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
        break
    time.sleep(5)
events = [json.loads(line) for line in (HERE/'native-reference/runs.jsonl').read_text().splitlines()]
assert any(e.get('kind')=='complete' for e in events), 'baseline did not finish'
assert events[-1]['kind'] == 'released'
run = Run(HERE, time.time()+300)
try:
    run.quiet()
    run.held = True
    run.renew(force=True)
    bench.gw('POST','/api/v1/gpu/free',{})
    assert not any(m>300 for m in run.gpu_apps().values()), 'foreign GPU use'
    with (HERE/'confidence-cuda.txt').open('w') as log:
        subprocess.run([str(HERE/'check_confidence'),'cuda'],
                       env={**os.environ,'LD_LIBRARY_PATH':str(HERE/'e2e/reference')},
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
finally:
    run.stop()
print('CONFIDENCE_CUDA_OK; starting aligned reference',flush=True)
with (HERE/'aligned-reference-run.txt').open('w') as log:
    subprocess.run([sys.executable,str(HERE/'native_reference.py'),'--engines','strata-reference',
                    '--out','aligned-reference','--minutes','25'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('ALIGNED_REFERENCE_COMPLETE',flush=True)
