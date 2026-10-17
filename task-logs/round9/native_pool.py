#!/usr/bin/env python3
"""Build the existing native CPU pool and prove the isolated library substitution."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    commands = [
        ['cmake', '-S', '.', '-B', 'build-cpu-native', '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DGGML_NATIVE=ON', '-DGGML_OPENMP=OFF',
         '-DGGML_CUDA=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
         '-DLLAMA_BUILD_SERVER=OFF', '-DBUILD_SHARED_LIBS=ON'],
        ['nice', '-n', '10', 'cmake', '--build', 'build-cpu-native', '--target', 'ggml-cpu', '-j', '16'],
    ]
    for i, cmd in enumerate(commands):
        with (OUT / f'native-build-{i}.log').open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
    library = (ROOT / 'build-cpu-native/bin/libggml-cpu.so.0').resolve()
    cmd = ['python3', '-S', 'task-logs/round9/quality.py', 'qwen', 'off', '--probe',
           '--tag', 'nativepool', '--cpu-library', str(library)]
    with (OUT / 'native-quality.log').open('w') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    control = next((OUT / 'qwen-off-probe').glob('*.logits.bin'))
    candidate = next((OUT / 'qwen-off-nativepool-probe').glob('*.logits.bin'))
    result = dict(build_commands=commands, quality_command=cmd, cpu_library=str(library),
                  cpu_library_sha256=sha(library), control_sha256=sha(control), candidate_sha256=sha(candidate),
                  bytes=control.stat().st_size, production_source_changed=False, runtime_installed=False)
    result['bit_exact'] = result['control_sha256'] == result['candidate_sha256']
    (OUT / 'native-pool-proof.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['bit_exact'], 'native pool failed disabled-cache parity'
    path = OUT / 'extra-configs.json'
    extra = json.loads(path.read_text())
    # libggml-base still links libgomp, whose binding can narrow inherited masks.
    # Use the same explicit physical-core placement as the OpenMP comparison.
    base = extra['qwen-cpu-t24-physical']
    extra['qwen-cpu-native-pool'] = dict(cmd=base['cmd'], env={**base['env'],
        'LD_PRELOAD': str(library), 'LD_LIBRARY_PATH': str(ROOT / 'build-cuda/bin')})
    path.write_text(json.dumps(extra, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
