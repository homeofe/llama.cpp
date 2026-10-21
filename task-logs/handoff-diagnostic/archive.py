#!/usr/bin/env python3
"""Retain artifact identities and diagnostic-only history rows."""
import csv
import hashlib
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
manifest = {'scope': 'Two diagnostic captures; no adoption benchmark', 'files': {}}
paths = [HERE / 'cpu_ranges.so', Path('/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb/engine/strata')]
for name in ('aligned', 'native'):
    paths.extend((HERE / name).glob('trace.*'))
runtime = ROOT / 'task-logs/round16/e2e/reference-packed-v2'
paths.extend(p for p in runtime.iterdir() if p.is_file() and not p.is_symlink())
for p in paths:
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    manifest['files'][str(p)] = {'bytes': p.stat().st_size, 'sha256': digest}
manifest['build_command'] = 'g++ -shared -fPIC -O2 -I ggml/include task-logs/handoff-diagnostic/cpu_ranges.cpp -o task-logs/handoff-diagnostic/cpu_ranges.so -ldl -l:libnvToolsExt.so.1'
(HERE / 'artifact-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
history = ROOT / 'tools/bench/run-history.csv'
with history.open(newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    existing = [r for r in reader if not r['campaign'].startswith('handoff-diagnostic/')]
new = []
for name in ('aligned', 'native'):
    out = HERE / name
    cfg = json.loads((out / 'llama-config.json').read_text())
    if name == 'native':
        cfg = json.loads((out / 'strata-config.json').read_text())
    cmd = cfg['cmd'] if name == 'aligned' else [cfg['exe']] + cfg['args']
    model_flag = '-m' if name == 'aligned' else '--native'
    env = dict(cfg.get('env', {}))
    if name == 'aligned':
        env['LD_LIBRARY_PATH'] = str(runtime)
    else:
        env.update(STRATA_BF16_TC='1', STRATA_NO_LARGEPAGES='1', STRATA_SELECT_SIMT='1')
    for line in (out / 'runs.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e['kind'] not in ('warm_request', 'profiled_request'):
            continue
        t = e['timings']
        note = {'benchmark_valid': False, 'quiet': e.get('quiet'), 'load_start':e.get('load_start'),
                'foreign_cores':e.get('foreign_cores'), 'sha256':e['sha256'], 'draft_n':t['draft_n'],
                'draft_accepted':t['draft_n_accepted'], 'config':str((out / ('llama-config.json' if name=='aligned' else 'strata-config.json')).relative_to(ROOT)),
                'reason':'diagnostic only' if e.get('quiet') is not False else 'diagnostic; foreign-load gate failed'}
        new.append({'ts':e['utc']+' UTC','commit':'1cb045cb' if name=='native' else '1fe81c557 (CUDA checkpoint; manifest identifies all components)',
                    'tps':e['decode_tok_s'],'pp':t['prompt_per_second'],'test':e['kind'],'hw':'thread;3960X;RTX2080Ti;sm75;CUDA12.0',
                    'ok':0,'campaign':'handoff-diagnostic/'+name,'note':json.dumps(note,sort_keys=True),
                    'n_gen':e['tokens'],'origin':name,'cmd':shlex.join(cmd),'args':shlex.join(cmd[1:]),
                    'model':cmd[cmd.index(model_flag)+1], 'build':str(runtime) if name=='aligned' else cfg['exe'],
                    'env':json.dumps(env,sort_keys=True)})
with history.open('w',newline='') as f:
    writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(existing+new)
print(json.dumps({'new_diagnostic_rows':len(new),'artifacts':len(manifest['files'])}))
