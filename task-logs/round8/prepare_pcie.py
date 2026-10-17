#!/usr/bin/env python3
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
logs=root/'task-logs/round8'
model=json.loads((logs/'model-inventory.json').read_text())['qwen38']['layers']
with (root/'task-logs/round7/long.jsonl').open() as f:
    window=next(b for b in map(json.loads,f) if b['n_tokens']==4 and b['positions'][0]>=72)
misses=[]
layers=[]
for route in window['layers']:
    il=route['layer']
    tensors=model[str(il)]
    ids=sorted({e for row in route['ids'] for e in row})
    # A fixed surrogate miss mask; this is not a measured cache state.
    missed=[e for e in ids if (il*131+e*71)%10<6]
    size=sum(t['expert_bytes'] for t in tensors)
    misses += [(il,e,size*sum(row.count(e) for row in route['ids']),size) for e in missed]
    layers.append(dict(layer=il,ids=route['ids'],missed=missed,tensors=tensors))
misses.sort(key=lambda x: ((x[0]*131+x[1]*71)%8191,x[0],x[1]))
total=sum(x[2] for x in misses)
splits={}
for pct in [0,10,20,30]:
    used=0
    chosen=[]
    unique=0
    for il,e,size,unique_size in misses:
        if used+size <= total*pct//100:
            chosen.append([il,e]); used+=size; unique+=unique_size
    splits[str(pct)]=dict(experts=chosen,bytes=used,percent=100*used/total,unique_bytes=unique)
manifest=dict(source='round7/long.jsonl, first four-token window at or after position 72; deterministic 60% surrogate misses',
              fraction_basis='expert triplet bytes per missed route, including repeated experts',
              miss_bytes=total,unique_miss_bytes=sum(x[3] for x in misses),splits=splits,layers=layers)
(logs/'pcie-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('miss MiB',total/2**20,'split percentages',[s['percent'] for s in splits.values()])
