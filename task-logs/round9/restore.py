#!/usr/bin/env python3
"""Restore the two temporary system changes after all measurements end."""
import json
from pathlib import Path
import subprocess
import time

import bench

OUT = Path(__file__).resolve().parent


def main():
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    assert not gpu.strip(), 'a compute process is still using the GPU'
    status = json.loads(bench.stage.gw('GET', '/api/v1/status'))
    hold = status.get('gpuHold')
    if hold and hold['reason'] == 'codex: round 9':
        bench.stage.gw('DELETE', '/api/v1/gpu/hold')
    (OUT / 'compaction-release.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')
    for _ in range(10):
        if (OUT / 'compaction-restored.json').exists():
            break
        time.sleep(1)
    restored = json.loads((OUT / 'compaction-restored.json').read_text())
    compaction = int(Path('/proc/sys/vm/compaction_proactiveness').read_text())
    assert compaction == restored['previous'] == 20
    if Path('/sys/module/amd_uncore').exists():
        subprocess.run(['sudo', '-n', 'modprobe', '-r', 'amd_uncore'], check=True)
    assert not Path('/sys/module/amd_uncore').exists()
    status = json.loads(bench.stage.gw('GET', '/api/v1/status'))
    record = dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                  compaction_proactiveness=compaction, amd_uncore_loaded=False,
                  gpu_compute_processes=[], gpu_hold=status.get('gpuHold'),
                  gateway_loaded=status['loaded'], active_requests=status['activeRequests'],
                  live_runtime_changed=False, settings_changed=False)
    (OUT / 'restoration.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
