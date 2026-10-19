#!/usr/bin/env python3
"""Replay the qwen38-bench8 routing trace through our adaptive-LFU policy model and variants.

The trace holds one stage_b pass (8 prompts x 256 tokens, temperature 0); stage_b measures three
rounds of the same prompts, so the trace is replayed back to back to model cross-round retention.
Policy model (llama-moe-cache.cpp): counts decay x0.5 every `decay` tokens; a miss is admitted
only if its count beats the coldest resident; at most `admit` admissions per ubatch, `prefill`
during the prompt burst. Slots = target cache bytes / expert bytes.
Prints per-request hit rates so the trajectory can be compared with Strata's logged one.
"""
import collections, heapq, json, sys

TRACE = '/opt/llm-gateway/data/moe-quant/traces/qwen38-bench8.jsonl'
N_EXPERTS = 512
N_LAYERS = 48

def load():
    reqs = []          # each request: list of per-token frozen id sets
    cur = None
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line)
            if r.get('draft'):
                continue
            nt = r.get('n_tokens', 0)
            if nt == 0 or 'layers' not in r or not r['layers']:
                continue
            if r.get('positions', [1])[0] == 0:
                if cur:
                    reqs.append(cur)
                cur = []
            for i in range(nt):
                ids = set()
                for lay in r['layers']:
                    if i < len(lay['ids']):
                        for e in lay['ids'][i]:
                            ids.add((lay['layer'], e))
                if ids:
                    cur.append(ids)
    if cur:
        reqs.append(cur)
    return reqs

class LFU:
    def __init__(self, slots, decay, admit, prefill_cap):
        self.slots, self.decay, self.admit, self.prefill_cap = slots, decay, admit, prefill_cap
        self.counts = collections.defaultdict(float)
        self.resident = {}
        self.heap = []          # (count_at_push, key) lazy
        self.tokens = 0
        self.hapub = [0, 0]

    def tick(self):
        if self.tokens and self.decay and self.tokens % self.decay == 0:
            for k in self.counts:
                self.counts[k] *= 0.5
            self.heap = [(self.counts[k], k) for k in self.resident]
            heapq.heapify(self.heap)
        self.tokens += 1

    def coldest(self):
        while self.heap:
            f, k = self.heap[0]
            if k in self.resident and f == self.counts[k]:
                return f
            heapq.heappop(self.heap)
            if k in self.resident:
                heapq.heappush(self.heap, (self.counts[k], k))
        return float('inf')

    def observe(self, ids, prefill):
        # one token: every routed expert is observed; misses become admission candidates
        cands = []
        for k in ids:
            self.counts[k] += 1.0
            if k in self.resident:
                self.hapub[0] += 1
                heapq.heappush(self.heap, (self.counts[k], k))
            else:
                self.hapub[1] += 1
                cands.append(k)
        # admission: best `admit` misses by count, subject to beating the coldest resident
        cap = self.prefill_cap if prefill else self.admit
        cands.sort(key=lambda k: -self.counts[k])
        admitted = 0
        for k in cands:
            if admitted >= cap:
                break
            if len(self.resident) < self.slots:
                self.resident[k] = True
                heapq.heappush(self.heap, (self.counts[k], k))
                admitted += 1
                continue
            f = self.coldest()
            if f >= self.counts[k]:
                continue
            # evict one coldest
            while self.heap:
                fc, kc = heapq.heappop(self.heap)
                if kc in self.resident and fc == self.counts[kc]:
                    del self.resident[kc]
                    break
                if kc in self.resident:
                    heapq.heappush(self.heap, (self.counts[kc], kc))
            self.resident[k] = True
            heapq.heappush(self.heap, (self.counts[k], k))
            admitted += 1
        return self.hapub

def run(slots, decay, admit, prefill_cap, rounds=3):
    reqs = load()
    lfu = LFU(slots, decay, admit, prefill_cap)
    traj = []
    for rd in range(rounds):
        for req in reqs:
            h = a = 0
            for ids in req:
                lfu.tick()
                before = tuple(lfu.hapub)
                lfu.observe(ids, False)
                h += lfu.hapub[0] - before[0]
                a += lfu.hapub[1] - before[1]
            traj.append((round(100 * h / (h + a), 1), h + a))
    return traj

if __name__ == '__main__':
    slots = int(2944 * 1024 * 1024 / 2176000)     # target cache bytes / expert bytes
    print('slots:', slots)
    configs = [
        ('ours  (decay128, admit32)', dict(slots=slots, decay=128, admit=32, prefill_cap=1024)),
        ('decay512',                   dict(slots=slots, decay=512, admit=32, prefill_cap=1024)),
        ('decay2048',                  dict(slots=slots, decay=2048, admit=32, prefill_cap=1024)),
        ('no-decay (decay=1e9)',      dict(slots=slots, decay=10**9, admit=32, prefill_cap=1024)),
        ('admit256',                   dict(slots=slots, decay=128, admit=256, prefill_cap=1024)),
        ('no-decay + admit256',        dict(slots=slots, decay=10**9, admit=256, prefill_cap=1024)),
        ('no-decay + admit64',         dict(slots=slots, decay=10**9, admit=64, prefill_cap=1024)),
    ]
    for name, kw in configs:
        traj = run(**kw)
        rates = [t[0] for t in traj]
        ups = [t[1] for t in traj]
        # cumulative over all 24 requests, and mean of the last 8 (steady state)
        tot_h = sum(rates[i] * ups[i] for i in range(len(rates))) / sum(ups)
        steady = sum(rates[-8:]) / 8
        print(f'{name:28s} cum {tot_h:5.1f}%  steady(last round) {steady:5.1f}%  '
              f'traj {" ".join(f"{r:.0f}" for r in rates)}')
