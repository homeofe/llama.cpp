#!/usr/bin/env python3
"""Summarize the Strata engine log of a stage1 run: per-request decode ms, drafts, expert hits."""
import json, re, sys
from pathlib import Path

REQ = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms.*?, (\d+) generated in '
                 r'([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)')
HIT = re.compile(r'decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)')
PCI = re.compile(r'PCIe')
KV = re.compile(r'KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM')
RAM = re.compile(r'(\d+) RAM blobs|ram_blobs')

log = Path(sys.argv[1])
rows = []
for line in log.read_text(errors='replace').splitlines():
    m = REQ.search(line)
    if m:
        rows.append(dict(prompt_tok=int(m[1]), reused=int(m[2]), read=int(m[3]), prompt_ms=float(m[4]),
                         generated=int(m[5]), decode_ms=float(m[6]), tok_s=float(m[7]),
                         drafts_acc=int(m[8]), drafts_off=int(m[9])))
    m = HIT.search(line)
    if m and rows:
        rows[-1]['hit_pct'] = float(m[1])
        rows[-1]['hits'] = int(m[2])
        rows[-1]['lookups'] = int(m[3])
    m = KV.search(line)
    if m and rows:
        rows[-1]['kv_vram_pct'] = float(m[1])
        rows[-1]['kv_blocks'] = int(m[2])
for r in rows:
    rounds = r['generated'] - r['drafts_acc']
    r['rounds'] = rounds
    r['ms_per_round'] = round(r['decode_ms'] / rounds, 2) if rounds else None
    r['tok_per_round'] = round(r['generated'] / rounds, 2) if rounds else None
    if 'hits' in r and rounds:
        r['cpu_experts_per_round'] = round((r['lookups'] - r['hits']) / rounds, 1)
        r['gpu_hits_per_round'] = round(r['hits'] / rounds, 1)
print(json.dumps(rows, indent=1))
