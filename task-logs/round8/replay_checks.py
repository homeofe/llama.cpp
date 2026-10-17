#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[2]
logs = root / 'task-logs/round8'
binary = root / 'build-cpu/bin/test-moe-cache'

def batch(pos, ids, prompt=False):
    return dict(version=1, context=1, ubatch=pos, n_tokens=len(ids), prefill=prompt, prefill_end=prompt,
                positions=list(range(pos,pos+len(ids))), seq_ids=[[0] for _ in ids], layers=[
                    dict(layer=0, group=0, slots=1, n_expert=4, expert_bytes=100, cache=not prompt, rows=list(range(len(ids))), ids=[[x] for x in ids]),
                    dict(layer=1, group=1, slots=1, n_expert=4, expert_bytes=200, cache=not prompt, rows=list(range(len(ids))), ids=[[0] for _ in ids])])

records = [batch(0,[0],True),
           dict(version=1,event='spec_begin',context=1,seq=0,pos=1,n_tokens=2,n_accepted=-1),
           batch(1,[1,1]),
           dict(version=1,event='spec_accept',context=1,seq=0,pos=1,n_tokens=2,n_accepted=1),
           batch(3,[1]),batch(4,[1])]
fixture = logs / 'replay-fixture.jsonl'
fixture.write_text(''.join(json.dumps(r)+'\n' for r in records))
for accepted, expected in [(0, (75.0,83.3333,0)), (1,(62.5,75.0,2))]:
    env = {**os.environ,'GGML_MOE_REPLAY_ACCEPTED_ONLY':str(accepted)}
    out = subprocess.check_output([str(binary),'--replay-log',str(fixture),'0','2','2','0','128'],text=True,env=env)
    (logs/f'replay-fixture-{accepted}.log').write_text(out)
    metrics = dict(re.findall(r'([\w/]+)=([^\s]+)',next(x for x in out.splitlines() if x.startswith('context='))))
    for key, value in zip(['decode_hit_rate','decode_byte_hit_rate','rejected_routes'], expected):
        assert abs(float(metrics[key].rstrip('%'))-value)<0.0001,(key,metrics,expected)
print('Multi-group weighting and causal rejected-route feedback passed')

def check_fixture(name, records, expected, admit=1, prefill=1, decay=128, accepted=0):
    path = logs / f'{name}.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    env = {**os.environ, 'GGML_MOE_REPLAY_ACCEPTED_ONLY':str(accepted),
           'GGML_MOE_HYBRID_BYTE_VALUE':'0', 'GGML_MOE_HYBRID_LAYER_SOFT_PCT':'0'}
    out = subprocess.check_output([str(binary),'--replay-log',str(path),'0',str(admit),str(prefill),'0',str(decay)],text=True,env=env)
    (logs/f'{name}.log').write_text(out)
    metrics = dict(re.findall(r'([\w/]+)=([^\s]+)',next(x for x in out.splitlines() if x.startswith('context='))))
    for key, value in expected.items():
        assert abs(float(metrics[key].rstrip('%'))-value)<0.0001,(name,key,metrics,expected)

def single(pos, ids, prompt=False):
    b = batch(pos, ids, prompt)
    b['layers'] = b['layers'][:1]
    return b

def event(seq, pos, nt, accepted=-1):
    return dict(version=2,event='spec_begin' if accepted<0 else 'spec_accept',
                context=1,seq=seq,pos=pos,n_tokens=nt,n_accepted=accepted)

check_fixture('replay-decayed-verdict', [single(0,[0],True),event(0,1,4),
              single(1,[1,1,2,2]),event(0,1,4,4),single(5,[2])],
              {'decode_hit_rate':20,'uploads':2,'unresolved_spans':0},decay=2,accepted=1)
mixed = single(0,[0,1,1])
mixed['positions'] = [0,0,1]
mixed['seq_ids'] = [[0],[1],[1]]
tail = single(2,[1])
tail['seq_ids'] = [[1]]
check_fixture('replay-verdict-cap', [event(0,0,1),event(1,0,2),mixed,
              event(0,0,1,1),event(1,0,2,2),tail],
              {'decode_hit_rate':0,'uploads':2,'unresolved_spans':0},prefill=0,accepted=1)

records = []
for pos, ids, prompt in [(0,[0],True),(1,[1,1],False),(3,[2,2,2],False),(6,[2],False)]:
    b = batch(pos,ids,prompt)
    b['n_groups'] = 3
    b['layers'][1].update(layer=2,group=2,ids=[[x] for x in ids])
    b['layers'].append(dict(layer=1,group=-1,slots=0,n_expert=4,expert_bytes=300,
                            cache=False,rows=list(range(len(ids))),ids=[[0] for _ in ids]))
    records.append(b)
check_fixture('replay-dropped-group',records,{'decode_byte_hit_rate':5.5556,'groups':2},prefill=2)
uncached = single(0,[0,1])
uncached['trace_only'] = True
uncached['layers'][0].update(group=-1,slots=0)
check_fixture('replay-uncached-context',[uncached],{'decode_byte_hit_rate':0,'groups':0,'uploads':0})
print('Delayed decay, per-ubatch verdict cap, and dropped-group rotation passed')

for work in ['short','long']:
    for decay in [64,128,512]:
        for prefill,static in [(0,0),(1024,0),(0,25),(1024,25)]:
            args = ['--replay-log',str(root/f'task-logs/round7/{work}.jsonl'),'1501','32',str(prefill),str(static),str(decay),str(root/'task-logs/round7/base.profile')]
            before = subprocess.check_output([str(logs/'baseline/test-moe-cache')]+args,text=True,env={**os.environ,'LD_LIBRARY_PATH':str(logs/'baseline')})
            after = subprocess.check_output([str(binary)]+args,text=True)
            a = next(x for x in before.splitlines() if x.startswith('context='))
            b = next(x for x in after.splitlines() if x.startswith('context='))
            assert b.startswith(a+' '),(work,decay,prefill,static,a,b)
            (logs/f'parity-{work}-{decay}-{prefill}-{static}.log').write_text(after)
print('All 24 round-7 replay rows match the saved round-7 binary')
