#!/usr/bin/env python3
"""Compare 8K and 32K with the measured cache budget and slot count."""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'
spec = importlib.util.spec_from_file_location('summary10', ROOT / 'task-logs/round10/summarize.py')
summary10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary10)


def main():
    fit = json.loads((OUT / 'context-fit.json').read_text())
    summary = summary10.summarize(OUT / 'context-e2e', 'ling-8k')
    assert summary['complete']
    directory = OUT / ('ling-32k-cache-fit-' + str(fit['cache_budget_mib']))
    text = (directory / (directory.name + '.server.log')).read_text()
    slots = re.findall(r'adaptive LFU: (\d+) slots', text)
    sizes = re.findall(r'CUDA0 MoE cache size =\s*([\d.]+) MiB', text)
    assert slots and sizes
    old = (OUT / 'bound/ling-admit32.server.log').read_text()
    original_slots = int(re.findall(r'adaptive LFU: (\d+) slots', old)[-1])
    fit.update(cache_slots=int(slots[-1]), original_cache_slots=original_slots,
        slots_lost=original_slots - int(slots[-1]), allocated_cache_mib=float(sizes[-1]))
    changes = []
    for category in ['code', 'english']:
        a = next(r for r in summary['summary'] if r['config'] == 'ling-8k' and r['category'] == category)
        b = next(r for r in summary['summary'] if r['config'] == 'ling-32k' and r['category'] == category)
        changes.append(dict(category=category, control_8k_tok_s=a['tok_s'], context_32k_tok_s=b['tok_s'],
            change_pct=(b['tok_s'] / a['tok_s'] - 1) * 100, same_text=b['text_matches_control'], pairs=b['requests']))
    result = dict(fit=fit, speed=summary, changes=changes,
        scope='One accepted eight-prompt pass per context. Short-prompt throughput is a context/cache tradeoff screen, not three-round code-adoption evidence. The VRAM fit also replayed the long-combine input with 16 output tokens; the full panel uses its original output budget.')
    (OUT / 'context-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'context-fit.json').write_text(json.dumps(fit, indent=2) + '\n')
    print(json.dumps(dict(fit=fit, changes=changes), indent=2))


if __name__ == '__main__':
    main()
