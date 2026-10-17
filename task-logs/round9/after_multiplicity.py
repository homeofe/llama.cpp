#!/usr/bin/env python3
"""Proceed to the CPU screen only when every Qwen miss type fails the kernel gate."""
import json
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    while not (OUT / 'cpu-decision-ready.txt').exists():
        if time.time() > 1792275900:  # 22:25 UTC; leave time for the bounded screen.
            raise TimeoutError('drafter and graph diagnostics did not finish in time')
        time.sleep(5)
    rows = json.loads((OUT / 'multiplicity.json').read_text())
    wanted = {'IQ4_NL', 'IQ3_XXS', 'IQ3_S', 'IQ2_S'}
    selected = [r for r in rows if r['trace'] == 'qwen38-bench8' and r['policy'] == 'r7'
                and r['phase'] == 'decode' and r['quant'] in wanted]
    assert {r['quant'] for r in selected} == wanted, 'incomplete real-trace multiplicity evidence'
    qualifies = [r['quant'] for r in selected if r['mean'] >= 1.6]
    decision = dict(threshold=1.6, selected_policy='r7', measurements=selected,
                    qualifying_types=qualifies, proceed_to_cpu_sweep=not qualifies,
                    reason='No Qwen type clears the specified multiplicity gate.' if not qualifies else
                           'Kernel design and correctness work must precede the CPU sweep.',
                    cpu_evidence='cpu-profile-summary.json: CPU graph work and substantial OpenMP spin waiting; graph durations can overlap GPU work.')
    (OUT / 'cpu-kernel-decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(dict(qualifying_types=qualifies, proceed_to_cpu_sweep=not qualifies)), flush=True)
    if qualifies:
        return 42
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%H:%M:%S') + ' Runde 9: Kein Qwen-IQ-Typ erreicht die Miss-Multiplizitaet 1,6; keine neue Multi-Token-Kernelfamilie. CPU-Thread-/Threadpool-Vergleich startet.\n')
    with (OUT / 'cpu-screen-driver.log').open('w') as stream:
        subprocess.run(['python3', '-u', 'task-logs/round9/cpu_screen.py'], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
