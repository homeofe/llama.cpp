#!/usr/bin/env python3
"""Negative cuts for missing attachment, one omitted CPU field and destructor order."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'
SOURCE = ROOT / 'common/speculative.cpp'


def run(command, tag):
    with (OUT / (tag + '.log')).open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'LD_LIBRARY_PATH': str(ROOT / 'build-cuda/bin')},
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            return process.wait(timeout=360)
        except BaseException:
            process.terminate()
            process.wait(timeout=70)
            raise


def build(tag):
    assert run(['nice', '-n', '10', 'cmake', '--build', 'build-cuda', '--target', 'llama-common', '-j', '16'], tag) == 0


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    assert json.loads((OUT / 'pool-same-model/verdict.json').read_text())['passed']
    original = SOURCE.read_text()
    results = []
    try:
        code = run(['python3', '-u', str(OUT / 'pool_runtime.py'), 'qwen', 'missing-attach', '--baseline'], 'pool-missing-attach-driver')
        verdict = json.loads((OUT / 'pool-missing-attach/verdict.json').read_text())
        assert code != 0 and verdict['attaches'] == 1 and verdict['create_counts'] == [1, 1]
        results.append(dict(cut='original library missing draft attachment', exit=code, verdict=verdict))

        needle = '        result.cpuparams       = params_spec.cpuparams;'
        assert original.count(needle) == 1
        SOURCE.write_text(original.replace(needle, needle + '\n        result.cpuparams.poll = 50;'))
        build('pool-tiny-cut-build')
        code = run([str(ROOT / 'build-cuda/bin/test-arg-parser')], 'pool-tiny-cut-args')
        assert code == -signal.SIGABRT
        results.append(dict(cut='omit draft poll field', exit=code))
        SOURCE.write_text(original)
        build('pool-tiny-cut-restored-build')
        assert run([str(ROOT / 'build-cuda/bin/test-arg-parser')], 'pool-tiny-cut-restored-args') == 0

        needle = '    common_threadpools threadpools;\n    llama_model_ptr   model;\n    llama_context_ptr context;'
        assert original.count(needle) == 1
        SOURCE.write_text(original.replace(needle, '    llama_model_ptr   model;\n    llama_context_ptr context;\n    common_threadpools threadpools;'))
        build('pool-order-cut-build')
        code = run(['python3', '-u', str(OUT / 'pool_runtime.py'), 'qwen', 'order-cut'], 'pool-order-cut-driver')
        verdict = json.loads((OUT / 'pool-order-cut/verdict.json').read_text())
        events = (OUT / 'pool-order-cut/events.txt').read_text()
        assert code != 0 and verdict['returncode'] == -signal.SIGABRT and 'R10_POOL violation' in events
        assert verdict['attaches'] == 2 and verdict['placement_ok']
        results.append(dict(cut='destroy pool before context', exit=code, verdict=verdict))
    finally:
        SOURCE.write_text(original)
        build('pool-restored-build')
        code = run([str(ROOT / 'build-cuda/bin/test-arg-parser')], 'pool-restored-args')
        results.append(dict(restored_source=True, arg_test_exit=code, epoch=time.time()))
        (OUT / 'pool-mutation-verdict.json').write_text(json.dumps(results, indent=2) + '\n')
        assert code == 0
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
