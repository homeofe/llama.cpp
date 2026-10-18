#!/usr/bin/env python3
"""Fresh numeric gates, then three interleaved full prompt passes."""
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'


def main():
    while not (OUT / 'pool-mutation-verdict.json').exists():
        if time.time() > 1792318800:
            raise TimeoutError('mutation checks did not finish by10:20 UTC')
        time.sleep(3)
    cuts = json.loads((OUT / 'pool-mutation-verdict.json').read_text())
    assert len(cuts) == 4 and cuts[-1]['arg_test_exit'] == 0
    library = ROOT / 'build-cpu-native/bin/libggml-cpu.so.0.27.0'
    assert hashlib.sha256(library.read_bytes()).hexdigest() == '32a87658688e5daa2dce82554dada6e5ec31b3fe6d5aafa1dcfa0d2eb363186c'
    jobs = [['off', '--probe'], ['off'], ['r7']]
    process = None
    results = []
    try:
        for index, flags in enumerate(jobs):
            command = ['python3', '-u', str(OUT / 'quality.py'), 'qwen', *flags, '--cpu-library', str(library),
                       '--tag', 'nativefixed', '--attempts', '2']
            with (OUT / f'pool-quality-{index}.log').open('w') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            code = process.wait()
            results.append(dict(command=command, exit=code, epoch=time.time()))
            (OUT / 'pool-pipeline-progress.json').write_text(json.dumps(results, indent=2) + '\n')
            assert code == 0, 'quality check failed'
        probe = next((OUT / 'qwen-off-nativefixed-probe').glob('*.logits.bin'))
        digest = hashlib.sha256(probe.read_bytes()).hexdigest()
        assert digest == 'b7facadb625ef4a18eb43b74959a4daa766b477a0dd2bb0bdeec6f0b984fd5a0'
        ppls = {}
        for policy in ['off', 'r7']:
            directory = OUT / f'qwen-{policy}-nativefixed-ppl'
            text = (directory / (directory / 'accepted.txt').read_text().strip()).read_text()
            match = re.search(r'Final estimate: PPL = ([\d.]+)', text)
            assert match, 'missing completed perplexity result'
            ppls[policy] = float(match[1])
        assert ppls['r7'] <= ppls['off'] * 1.005
        (OUT / 'pool-quality-verdict.json').write_text(json.dumps(dict(cache_off_bit_exact=True, sha256=digest,
            bytes=probe.stat().st_size, ppl=ppls, change_pct=100*(ppls['r7']/ppls['off']-1)), indent=2) + '\n')
        with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
            stream.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + ' Round10: Native Threadpools bestehen Logitparitaet und PPL; drei verschachtelte Qwen-Vergleichsrunden starten.\n')
        command = ['python3', '-u', str(OUT / 'pool_bench.py'), str(OUT / 'pool-results'), '10:55',
                   '--only', 'qwen-pool-control', 'qwen-pool-native', '--rounds', '3']
        with (OUT / 'pool-bench-driver.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        code = process.wait()
        results.append(dict(command=command, exit=code, epoch=time.time()))
        events = [json.loads(x) for x in (OUT / 'pool-results/runs.jsonl').read_text().splitlines()]
        assert code == 0 and any(x['kind'] == 'all_complete' for x in events)
        assert len([x for x in events if x['kind'] == 'request' and x['accepted']]) == 48
        (OUT / 'pool-pipeline-complete.json').write_text(json.dumps(results, indent=2) + '\n')
        print('Pool quality and three benchmark rounds complete', flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=70)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
