#!/usr/bin/env python3
"""Export every round-17 request; diagnostics and invalid comparisons stay excluded."""
import csv
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
fields = 'ts,commit,bno,tps,pp,hit,model,args,test,hw,ok,build,md5,campaign,note,ppl,spp,n_gen,origin,env,cmd'.split(',')
rows = []
for campaign in ('property-probe', 'aligned-pilot', 'cap2-probe', 'aligned-cap2', 'cpu-aligned'):
    out = HERE / campaign
    config_file = out / ('configs.json' if (out / 'configs.json').exists() else 'config.json')
    configs = json.loads(config_file.read_text())
    if 'config' in configs:
        configs = configs['config']
    events = [json.loads(line) for line in (out / 'runs.jsonl').read_text().splitlines()]
    complete = {(e['block'], e['attempt']) for e in events if e['kind'] == 'block_complete'}
    for e in events:
        if e['kind'] != 'request':
            continue
        cfg = configs[e['config']] if 'cmd' not in configs else configs
        cmd = cfg['cmd']
        runtime = Path(cmd[0]).resolve().parent
        valid = (campaign == 'aligned-cap2' and e['accepted'] and (e['block'], e['attempt']) in complete)
        reasons = []
        if campaign != 'aligned-cap2':
            reasons.append('invalid RUNPATH pilot' if campaign == 'aligned-pilot' else 'diagnostic, not benchmark')
        if campaign == 'aligned-cap2' and e['prompt'] == 'en-1' and e['phase'] == 'repeat':
            valid = False
            reasons.append('clean request; excluded from isolated patch comparison: differing adaptive draft counts')
        timing = e['timings']
        env = dict(cfg['env'])
        if campaign in ('aligned-cap2', 'cap2-probe', 'cpu-aligned'):
            env['LD_LIBRARY_PATH'] = str(runtime)
        source = 'edbd6a4ab' if campaign == 'cap2-probe' or (campaign == 'aligned-cap2' and e['config'] == 'B') else '1fe81c557'
        if campaign == 'property-probe':
            source = '2727f948f'
        if campaign == 'aligned-pilot':
            source = 'unverified-build-libraries'
        note = dict(reasons=reasons, source_scope='CUDA code checkpoint; all component hashes in round17 manifests',
                    output_sha256=e['sha256'], draft_n=timing['draft_n'], draft_accepted=timing['draft_n_accepted'],
                    foreign_cores=e.get('foreign_cores'), load_start=e.get('load_start'), gpu_mib=e.get('gpu'),
                    evidence=str((out / 'runs.jsonl').relative_to(ROOT)))
        rows.append(dict(ts=e['utc'] + ' UTC', commit=source, tps=f"{e['decode_tok_s']:.6f}",
                         pp=timing['prompt_per_second'], model=cmd[cmd.index('-m') + 1], args=shlex.join(cmd[1:]),
                         test=e.get('tag') or (e['prompt'] + '/' + e['phase']), hw='thread;3960X;RTX2080Ti;sm75;CUDA12.0',
                         ok=int(valid), build=str(runtime.relative_to(ROOT)), campaign='round17/' + campaign,
                         note=json.dumps(note, separators=(',', ':')), n_gen=e['tokens'], origin='fork',
                         env=json.dumps(env, sort_keys=True), cmd=shlex.join(cmd)))
rows.append(dict(ts='2026-10-21 13:42:48 UTC', commit='unverified-build-libraries', ok=0,
                 campaign='round17/aligned-pilot', test='capacity4/OOM', origin='fork',
                 note='Candidate aborted in cudaGraphInstantiate before any measured response; entire campaign excluded.'))
target = ROOT / 'tools/bench/run-history.csv'
target.parent.mkdir(parents=True, exist_ok=True)
previous = []
if target.exists():
    with target.open(newline='') as f:
        previous = [row for row in csv.DictReader(f) if not row['campaign'].startswith('round17/')]
with target.open('w', newline='') as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(previous + rows)
print(json.dumps(dict(rows=len(rows),eligible=sum(int(r['ok']) for r in rows),path=str(target))))
