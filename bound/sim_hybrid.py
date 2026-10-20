#!/usr/bin/env python3
"""Hybrid admission policy simulator: LFU with decay + an evict-protected pinned top-K set.

The pin set is learned in a first pass (the all-time counts of the same trace), mirroring a
profile file learned from the deployed workload. The pin-0 configuration must reproduce the
plain per-token LFU (43.7% cumulative at 1779 slots / decay 128 / admit 32) - that reference
check is the gate before any pin result is believed.

Report, separately: the cumulative/steady hit rate AND the round-1 (novel requests) average -
a policy that games the benchmark's repeated rounds while losing novel-request performance is
not adoptable.
"""
import sys, collections, heapq

sys.path.insert(0, '.')
import sim_policy as sp


class PinLFU:
    def __init__(self, slots, decay, admit, pin):
        self.slots, self.decay, self.admit = slots, decay, admit
        self.pin = pin                       # set of never-evicted keys
        self.counts = collections.defaultdict(float)
        self.resident = set()
        self.n_pin_res = 0                   # pinned residents
        self.heap = []                       # (count, key) of NON-pinned residents, lazy
        self.tokens = 0
        self.hapub = [0, 0]

    def tick(self):
        if self.tokens and self.tokens % self.decay == 0:
            for k in self.counts:
                self.counts[k] *= 0.5
            self.heap = [(self.counts[k], k) for k in self.resident if k not in self.pin]
            heapq.heapify(self.heap)
        self.tokens += 1

    def _pop_coldest_nonpin(self):
        """True coldest non-pinned resident, or None (lazy heap, skip stale)."""
        while self.heap:
            f, k = heapq.heappop(self.heap)
            if k in self.resident and k not in self.pin and f == self.counts[k]:
                return f, k
            if k in self.resident and k not in self.pin:
                heapq.heappush(self.heap, (self.counts[k], k))
        return None

    def observe(self, ids):
        cands = []
        for k in ids:
            self.counts[k] += 1.0
            if k in self.resident:
                self.hapub[0] += 1
                if k not in self.pin:
                    heapq.heappush(self.heap, (self.counts[k], k))
            else:
                self.hapub[1] += 1
                cands.append(k)
        # pinned candidates first (they may evict), then everyone else by count
        cands.sort(key=lambda k: (k not in self.pin, -self.counts[k]))
        admitted = 0
        for k in cands:
            if admitted >= self.admit:
                break
            if k in self.resident:
                continue
            if len(self.resident) < self.slots:
                self.resident.add(k)
                if k in self.pin:
                    self.n_pin_res += 1
                else:
                    heapq.heappush(self.heap, (self.counts[k], k))
                admitted += 1
                continue
            victim = self._pop_coldest_nonpin()
            if victim is None:
                break                           # only pinned residents left
            f, vk = victim
            if k not in self.pin and f >= self.counts[k]:
                heapq.heappush(self.heap, (f, vk))   # put it back, nothing to do
                break
            self.resident.discard(vk)
            self.resident.add(k)
            if k not in self.pin:
                heapq.heappush(self.heap, (self.counts[k], k))
            admitted += 1


def run(pin_n, slots=1779, decay=128, admit=32, rounds=3, trace=None):
    reqs = trace if trace is not None else sp.load()
    pin = set()
    if pin_n:
        all_time = collections.Counter()
        for req in reqs:
            for ids in req:
                all_time.update(ids)
        pin = {k for k, _ in all_time.most_common(pin_n)}
    lfu = PinLFU(slots, decay, admit, pin)
    traj = []
    for rd in range(rounds):
        for req in reqs:
            h = a = 0
            for ids in req:
                lfu.tick()
                b = tuple(lfu.hapub)
                lfu.observe(ids)
                h += lfu.hapub[0] - b[0]
                a += lfu.hapub[1] - b[1]
            traj.append((round(100 * h / max(1, h + a), 1), a))
    rates = [t[0] for t in traj]
    ups = [t[1] for t in traj]
    cum = sum(rates[i] * ups[i] for i in range(len(rates))) / sum(ups)
    n = len(reqs)
    r1 = sum(rates[:n]) / n
    steady = sum(rates[-n:]) / n
    return cum, r1, steady, rates


if __name__ == '__main__':
    for pin in (0, 150, 300, 600, 900, 1200):
        cum, r1, steady, rates = run(pin)
        print(f'pin {pin:4d}: cum {cum:5.1f}%  novel(round1) {r1:5.1f}%  steady(last round) {steady:5.1f}%')
        if pin in (0, 600):
            print(f'           traj: {[round(x, 1) for x in rates]}')
