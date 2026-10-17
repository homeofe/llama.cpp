#!/usr/bin/env python3
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
data=json.loads((root/'task-logs/round8/model-inventory.json').read_text())

def slots(model, mib):
    groups=data[model]['groups']
    total=sum(g['expert_bytes']*len(g['layers'])*g['tensors'][0]['shape'][2] for g in groups)
    used=next(v for k,v in data[model]['metadata'].items() if k.endswith('expert_used_count'))
    out=[]
    allocated=0
    for g in groups:
        ne=g['tensors'][0]['shape'][2]
        budget=int(mib*2**20*(g['expert_bytes']*len(g['layers'])*ne)/total)
        def size(n):
            pad=lambda v:(v+127)//128*128
            return len(g['layers'])*pad(ne*4)+sum(pad(t['expert_bytes']*(n+1)) for t in g['tensors'])
        n=0
        while n<len(g['layers'])*ne and size(n+1)<=budget:
            n+=1
        if n<min(8*used,ne):
            n=0
        out.append(n)
        allocated+=size(n) if n else 0
    return out,allocated/2**20

rows=[]
for model,budgets in [('qwen38',[2560,2944,3072]),('qwen38_mtp',[512]),('ling',[5120,5376])]:
    for budget in budgets:
        counts,mib=slots(model,budget)
        rows.append(dict(model=model,budget_mib=budget,slots=counts,total_slots=sum(counts),allocated_mib=mib))
        print(model,budget,'slots',counts,'total',sum(counts),'allocated MiB',round(mib,3))
(root/'task-logs/round8/slot-budget.json').write_text(json.dumps(rows,indent=2)+'\n')
