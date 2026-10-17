#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[2]
logs = root / 'task-logs/round8'
model = json.loads((logs/'model-inventory.json').read_text())['qwen38']
budget = next(r for r in json.loads((logs/'slot-budget.json').read_text()) if r['model']=='qwen38' and r['budget_mib']==2560)
layer_groups = {il:(ig,g) for ig,g in enumerate(model['groups']) for il in g['layers']}
rows=[]
policies=[('r7',128,32,1024,0,0),('decay64',64,32,1024,0,0),('prefill512',128,32,512,0,0),
          ('admit16',128,16,1024,0,0),('admit64',128,64,1024,0,0),('byte-value',128,32,1024,1,0),
          ('byte-decay64',64,32,1024,1,0),('byte-soft100',128,32,1024,1,100),
          ('byte-soft150',128,32,1024,1,150),('byte-prefill512',128,32,512,1,0)]
for workload in ['long','short']:
    trace=logs/f'legacy-layout-{workload}.jsonl'
    with (root/f'task-logs/round7/{workload}.jsonl').open() as source, trace.open('w') as dest:
        for line in source:
            b=json.loads(line)
            b['n_groups']=len(model['groups'])
            for layer in b['layers']:
                ig,g=layer_groups[layer['layer']]
                nslots=budget['slots'][ig]
                layer.update(n_expert=512,group=ig if nslots else -1,slots=nslots,expert_bytes=g['expert_bytes'])
                if not nslots:
                    layer['cache']=False
            dest.write(json.dumps(b,separators=(',',':'))+'\n')
    for name,decay,admit,prefill,byte,soft in policies:
        env={**os.environ,'GGML_MOE_HYBRID_BYTE_VALUE':str(byte),'GGML_MOE_HYBRID_LAYER_SOFT_PCT':str(soft)}
        out=subprocess.check_output([str(root/'build-cpu/bin/test-moe-cache'),'--replay-log',str(trace),'0',str(admit),str(prefill),'0',str(decay)],text=True,env=env)
        (logs/f'layout-{workload}-{name}.log').write_text(out)
        metric=dict(re.findall(r'([\w/]+)=([^\s]+)',next(l for l in out.splitlines() if l.startswith('context='))))
        rows.append(dict(workload=workload,policy=name,**metric))
(logs/'legacy-layout-matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
with (logs/'legacy-layout-table.md').open('w') as f:
    f.write('| Old workload, actual Qwen layouts | Policy | Decode bytes hit | First-32 bytes hit | MiB/token |\n| --- | --- | ---: | ---: | ---: |\n')
    for r in rows:
        f.write(f"| {r['workload']} | {r['policy']} | {r['decode_byte_hit_rate']} | {r['first32_byte_hit_rate']} | {r['MiB/token']} |\n")
print('Legacy-route mixed-layout replay completed; these are not real server traces')
