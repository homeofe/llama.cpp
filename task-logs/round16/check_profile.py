#!/usr/bin/env python3
"""Check packed versus proportional cache allocation, plus real-loader rejection controls."""
import json
import os
import signal
import time

from native_reference import Reference
from run_drafts import HERE, bench, configs

out = HERE/'profile-controls'
out.mkdir(exist_ok=False)
source = configs()['strata-reference']
skip = {'--spec-type','-md','--spec-draft-n-max','--spec-draft-n-ubatch','--spec-draft-ngl','--spec-draft-p-min'}
cmd = []
i = 0
while i < len(source['cmd']):
    arg = source['cmd'][i]
    if arg in skip:
        i += 2
    else:
        cmd.append(arg)
        i += 1
cmd += ['-lv','4']
profile = (HERE/'strata-target.profile').read_text()
bad = out/'duplicate.profile'
bad.write_text(profile + profile.splitlines()[1]+'\n')
run = Reference(out,time.time()+360)
def stop(*_):
    raise KeyboardInterrupt('SIGTERM')
signal.signal(signal.SIGTERM,stop)
results = {}
try:
    for name in ['packed','proportional','duplicate','budget']:
        cfg = dict(cmd=list(cmd),env=dict(source['env']))
        if name == 'proportional':
            cfg['env'].update(GGML_MOE_HYBRID_PACK_PROFILE='0', GGML_MOE_HYBRID_MIB='4032')
        if name == 'duplicate':
            cfg['env']['GGML_MOE_HYBRID_PROFILE'] = str(bad)
        if name == 'budget':
            cfg['env']['GGML_MOE_HYBRID_MIB'] = '1'
        (out/(name+'-config.json')).write_text(json.dumps(cfg,indent=2)+'\n')
        run.event(kind='correctness_only',case=name,benchmark_valid=False,load=os.getloadavg())
        try:
            run.start(name,cfg)
            assert name in ['packed','proportional'], 'invalid profile loaded'
            result = run.request(bench.PROMPTS[0][1],name,64)
            results[name] = result
            run.event(kind='integration_request',case=name,benchmark_valid=False,**result)
        except RuntimeError:
            expected = {'duplicate':'invalid or duplicate packed MoE profile expert',
                        'budget':'packed MoE profile exceeds device budget'}.get(name)
            assert expected and expected in (out/(name+'.server.txt')).read_text(), 'unexpected loader failure'
            results[name] = dict(rejected=True,error=expected)
        finally:
            run.stop()
    assert results['packed']['sha256'] == results['proportional']['sha256']
    assert results['packed']['tokens'] == results['proportional']['tokens']
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    run.event(kind='complete',output_identical=True,negative_controls=2)
finally:
    run.stop()
    run.event(kind='released',gpu=run.gpu_apps())
