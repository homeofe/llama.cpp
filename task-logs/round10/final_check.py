#!/usr/bin/env python3
"""Check final ownership, unchanged live files and restored experimental binaries."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import bench

ROOT = bench.ROOT
OUT = ROOT / 'task-logs/round10'


def main():
    initial = json.loads((OUT / 'initial-state.json').read_text())
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in initial['baseline_files']}
    assert hashes == initial['baseline_files']
    assert json.loads((OUT / 'pool-final-state.json').read_text())['original_source_and_runtime_restored']
    subprocess.run(['git', 'diff', '--exit-code', '0cc00dacd', '--', 'src', 'ggml', 'common', 'tools/server',
                    'tests/test-arg-parser.cpp'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    assert not apps.strip(), 'GPU compute process remains'
    status = json.loads(bench.stage.gw('GET', '/api/v1/status'))
    if status.get('gpuHold') and status['gpuHold']['reason'] == 'codex: round 10':
        bench.stage.gw('DELETE', '/api/v1/gpu/hold')
        status = json.loads(bench.stage.gw('GET', '/api/v1/status'))
    assert not status.get('gpuHold')
    assert not status['loaded'] and not status['activeRequests']
    active = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            argv = path.read_bytes().decode(errors='replace').split('\0')
        except (PermissionError, ProcessLookupError, FileNotFoundError):
            continue
        if any((arg.startswith(str(OUT) + '/') or arg.startswith('task-logs/round10/')) and arg.endswith('.py') for arg in argv):
            active.append(dict(pid=int(path.parent.name), argv=argv))
    assert not active, active
    capture_pids = set()
    for path in OUT.glob('profile-*/runs.jsonl'):
        for line in path.read_text().splitlines():
            capture_pids.update(json.loads(line).get('owned_profiler_processes', []))
    remaining = [pid for pid in capture_pids if Path(f'/proc/{pid}').exists()]
    assert not remaining, 'recorded profiler PID still exists; inspect ownership before cleanup'
    system = {key: Path(path).read_text().strip() for key, path in {
        'compaction_proactiveness': '/proc/sys/vm/compaction_proactiveness',
        'perf_event_paranoid': '/proc/sys/kernel/perf_event_paranoid',
        'ptrace_scope': '/proc/sys/kernel/yama/ptrace_scope'}.items()}
    assert system == dict(compaction_proactiveness='20', perf_event_paranoid='4', ptrace_scope='1')
    history = Path('/opt/llm-gateway/data/bench/results.jsonl')
    start = datetime.fromisoformat(initial['started_utc'].replace('Z', '+00:00')).timestamp()
    record = dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), baseline_file_hashes=hashes,
        original_source_restored=True, original_binary_hashes_verified=True,
        gpu_compute_processes=[], gpu_hold=status.get('gpuHold'), gateway_loaded=status['loaded'],
        active_requests=status['activeRequests'], active_round10_scripts=active, remaining_capture_pids=remaining,
        system=system, amd_uncore_loaded=Path('/sys/module/amd_uncore').exists(),
        cpu_governors=sorted({p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')}),
        live_history=dict(bytes=history.stat().st_size, mtime_epoch=history.stat().st_mtime,
                          modified_during_round=history.stat().st_mtime > start))
    (OUT / 'final-state.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
