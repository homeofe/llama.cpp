#!/usr/bin/env python3
"""Keep only a qualified experiment; otherwise restore the measured original files."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'


def main():
    assert (OUT / 'pool-pipeline-complete.json').exists()
    decision = json.loads((OUT / 'pool-decision.json').read_text())
    assert decision['complete']
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().split(b'\0')
        except (PermissionError, ProcessLookupError, FileNotFoundError):
            continue
        assert not any(str(directory / 'llama-server').encode() in command for directory in
                       [ROOT / 'build-cuda/bin', OUT / 'pool-baseline-bin', OUT / 'pool-candidate-bin']), 'owned model server still running'
    if not decision['adopt']:
        for name in ['common/speculative.cpp', 'tests/test-arg-parser.cpp']:
            (ROOT / name).write_bytes(subprocess.check_output(['git', 'show', '0cc00dacd:' + name], cwd=ROOT))
        subprocess.run(['cp', '-a', '--remove-destination', str(OUT / 'pool-baseline-bin') + '/.',
                        str(ROOT / 'build-cuda/bin') + '/'], check=True)
        expected = json.loads((OUT / 'pool-baseline-sha256.json').read_text())
        actual = {name: hashlib.sha256((ROOT / 'build-cuda/bin' / name).read_bytes()).hexdigest() for name in expected}
        assert expected == actual, 'original runtime restoration failed'
        (OUT / 'pool-restored-original-binaries.json').write_text(json.dumps(actual, indent=2) + '\n')
    (OUT / 'pool-final-state.json').write_text(json.dumps(dict(adopted_in_worktree=decision['adopt'],
        original_source_and_runtime_restored=not decision['adopt'], candidate_artifacts_retained=True,
        live_runtime_changed=False, epoch=time.time()), indent=2) + '\n')


if __name__ == '__main__':
    main()
