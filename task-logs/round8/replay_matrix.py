#!/usr/bin/env python3
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parents[2]
logs=root/'task-logs/round8'
rows=[]
policies=[('r7',128,32,1024,0,0)]
policies += [(f'd{d}-a{a}-p{p}',d,a,p,0,0) for d,a,p in itertools.product([64,128,256],[16,32,64],[256,512,1024]) if (d,a,p)!=(128,32,1024)]
policies += [(f'bytes-soft{s}',128,32,1024,1,s) for s in [0,100,150,200]]
for workload in ['long','short']:
    for name,decay,admit,prefill,byte,soft in policies:
        env={**os.environ,'GGML_MOE_HYBRID_BYTE_VALUE':str(byte),'GGML_MOE_HYBRID_LAYER_SOFT_PCT':str(soft)}
        out=subprocess.check_output([str(root/'build-cpu/bin/test-moe-cache'),'--replay-log',str(root/f'task-logs/round7/{workload}.jsonl'),'1501',str(admit),str(prefill),'0',str(decay)],text=True,env=env)
        (logs/f'matrix-{workload}-{name}.log').write_text(out)
        metric=dict(re.findall(r'([\w/]+)=([^\s]+)',next(l for l in out.splitlines() if l.startswith('context='))))
        rows.append(dict(workload=workload,policy=name,**metric))
(logs/'replay-matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
with (logs/'replay-table.md').open('w') as f:
    f.write('| Workload | Policy | Decode bytes hit | First-32 bytes hit | MiB/token |\n| --- | --- | ---: | ---: | ---: |\n')
    for r in rows:
        f.write(f"| {r['workload']} | {r['policy']} | {r['decode_byte_hit_rate']} | {r['first32_byte_hit_rate']} | {r['MiB/token']} |\n")
print(f'{len(rows)} replay rows completed; real traces unavailable')
