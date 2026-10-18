#!/usr/bin/env python3
"""Prove deferred upload ordering, including negative source mutations."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'
SOURCE = ROOT / 'src/llama-moe-cache.cpp'
TEST = ROOT / 'build-cpu/bin/test-moe-cache'
MODEL = ROOT / 'task-logs/round7/models/qwen3moe-moe.gguf'


def run(command, tag, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_ARG_')) and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH']}
    env.update(LD_LIBRARY_PATH=str(ROOT / 'build-cpu/bin'), TMPDIR=str(OUT / 'tmp'))
    env.update(extra or {})
    with (OUT / (tag + '.log')).open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            return process.wait(timeout=600)
        except BaseException:
            process.terminate()
            process.wait(timeout=60)
            raise


def build(tag, target='llama'):
    assert run(['nice', '-n', '10', 'cmake', '--build', 'build-cpu', '--target', target, '-j', '16'], tag) == 0


def main():
    (OUT / 'tmp').mkdir(exist_ok=True)
    assert json.loads((OUT / 'bound-summary.json').read_text())['complete']
    original = SOURCE.read_text()
    baseline = OUT / 'baseline-cpu-libllama.so'
    if not baseline.exists():
        shutil.copy2(ROOT / 'build-cpu/bin/libllama.so', baseline)
    build('host-build', 'test-moe-cache')
    command = [str(TEST), '--async-ordering', str(MODEL)]
    results = []
    positive = run(command, 'host-positive')
    results.append(dict(check='two-stream positive', exit=positive))
    assert positive == 0
    negative = run(command, 'host-old-library', dict(LD_PRELOAD=str(baseline)))
    results.append(dict(check='original library lacks async protocol', exit=negative))
    assert negative == 86
    cuts = [
        ('publish-before-completion', 'if (!d.uploading || !d.event_query(d.uploads_done.get())) {', 'if (!d.uploading) {', 'map(0)[0] == -1'),
        ('reuse-before-readers', 'ggml_backend_event_wait(d.upload_backend.get(), d.readers_done.get());', '/* mutation: no reader dependency */', '!queued.copies.empty()'),
        ('tiny-map-publication', 'ggml_backend_tensor_set_async(d.backend, layers[il].slot_map_device, &g.slot_values[f.slot + 1],', 'ggml_backend_tensor_set_async(d.backend, layers[il].slot_map_device, &g.slot_values[0],', 'map(il)[e] >= 0'),
    ]
    try:
        for name, needle, replacement, diagnostic in cuts:
            assert original.count(needle) == 1
            SOURCE.write_text(original.replace(needle, replacement))
            build('host-' + name + '-build')
            code = run(command, 'host-' + name)
            log = (OUT / ('host-' + name + '.log')).read_text(errors='replace')
            results.append(dict(check=name, exit=code, expected_diagnostic=diagnostic, diagnostic_found=diagnostic in log))
            assert code == 86 and diagnostic in log
    finally:
        SOURCE.write_text(original)
        build('host-restored-build')
        code = run(command, 'host-restored')
        results.append(dict(check='restored positive', exit=code))
        (OUT / 'host-verdict.json').write_text(json.dumps(results, indent=2) + '\n')
        assert code == 0
    assert run([str(TEST), '--ordering', str(MODEL), str(ROOT / 'task-logs/round7/warm.profile'), str(OUT / 'host-trace.jsonl')], 'host-legacy-ordering') == 0
    assert run([str(TEST)], 'host-policy') == 0
    sanitizer = OUT / 'asan-test-moe-cache'
    compile_command = ['nice', '-n', '10', 'g++', '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        '-DGGML_BACKEND_SHARED', '-DGGML_SHARED', '-DGGML_USE_CPU', '-DLLAMA_SHARED',
        '-Iinclude', '-Iggml/include', '-Icommon', '-Ivendor', '-Isrc',
        'tests/test-moe-cache.cpp', 'src/llama-moe-cache.cpp', '-Lbuild-cpu/bin',
        '-Wl,--export-dynamic', '-Wl,-rpath,' + str(ROOT / 'build-cpu/bin'),
        '-lllama-common', 'build-cpu/common/libllama-common-base.a', '-lllama', '-lggml', '-lggml-cpu', '-lggml-base',
        '-pthread', '-ldl', '-o', str(sanitizer)]
    assert run(compile_command, 'host-sanitizer-build') == 0
    assert run([str(sanitizer), '--async-ordering', str(MODEL)], 'host-sanitizer',
        dict(ASAN_OPTIONS='detect_leaks=1:abort_on_error=0:exitcode=87', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1:exitcode=88')) == 0
    (OUT / 'host-complete.json').write_text(json.dumps(dict(passed=True, epoch=time.time(),
        sanitizer_scope='Changed cache implementation and test executable instrumented with ASan/UBSan; existing CPU/model libraries unchanged.',
        sanitizer_command=compile_command), indent=2) + '\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
