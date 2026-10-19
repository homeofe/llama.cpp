#!/usr/bin/env python3
"""Analyze a Nsight Systems sqlite export: bound the last request burst, split it per round.

usage: analyze_trace.py TRACE.sqlite [--rounds N] [--out report.json]
The profiled request is the last activity burst; the warm-up request and model load
precede it, separated by quiet gaps (the harness's quiet gate guarantees seconds of idle).
"""
import argparse, json, sqlite3, sys
from pathlib import Path


def tables(con):
    return [r[0] for r in con.execute("select name from sqlite_master where type='table'")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--request-seconds', type=float, default=None, help='wall seconds of the request, to pick the burst')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    con = sqlite3.connect(a.trace)
    print('tables:', ', '.join(sorted(tables(con))))
    # kernels and memcpys
    kern = list(con.execute(
        'select k.start, k.end, s.value, k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k '
        'join StringIds s on s.id = k.shortName'))
    mc = list(con.execute(
        'select m.start, m.end, m.bytes, m.copyKind, m.streamId from CUPTI_ACTIVITY_KIND_MEMCPY m'))
    print(f'kernels {len(kern)}  memcpys {len(mc)}')
    if not kern:
        sys.exit('no kernels')
    kinds = {1: 'HtoD', 2: 'DtoH', 8: 'DtoD', 3: 'DtoHp', 10: 'HtoDp'}
    evs = [(s, e, 'K', name, stream) for s, e, name, stream in kern] + \
          [(s, e, 'M', (kinds.get(kind, str(kind)), nbytes), stream) for s, e, nbytes, kind, stream in mc]
    evs.sort()
    t0_all, t1_all = evs[0][0], max(e for _, e, *_ in evs)
    print(f'span {(t1_all - t0_all) / 1e9:.1f} s')
    # burst detection: gaps > 2 s split bursts; the last burst is the request
    bursts, cur = [], [evs[0]]
    for ev in evs[1:]:
        if ev[0] - cur[-1][1] > 2e9:
            bursts.append(cur)
            cur = [ev]
        else:
            cur.append(ev)
    bursts.append(cur)
    print('bursts:', [(round((b[0][0] - t0_all) / 1e9, 1), round((b[-1][1] - b[0][0]) / 1e9, 2), len(b))
                      for b in bursts])
    req = bursts[-1]
    w0, w1 = req[0][0], req[-1][1]
    if a.request_seconds and (w1 - w0) / 1e9 < a.request_seconds * 0.5:
        sys.exit(f'last burst only {(w1 - w0) / 1e9:.1f} s, expected about {a.request_seconds} s')
    rep = dict(trace=str(a.trace), window_s=round((w1 - w0) / 1e9, 3),
               bursts=[dict(offset_s=round((b[0][0] - t0_all) / 1e9, 1), dur_s=round((b[-1][1] - b[0][0]) / 1e9, 2)) for b in bursts])
    # per-kernel totals within the window
    agg = {}
    for s, e, kind, name, stream in req:
        if kind != 'K':
            continue
        d = agg.setdefault(name, [0, 0.0])
        d[0] += 1
        d[1] += (e - s)
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:40]
    rep['kernels'] = [dict(name=n, count=v[0], ms=round(v[1] / 1e6, 3)) for n, v in top]
    rep['kernel_total_ms'] = round(sum(v[1] for v in agg.values()) / 1e6, 3)
    # memcpy split
    kinds = {1: 'HtoD', 2: 'DtoH', 8: 'DtoD', 10: 'HtoDpinned?'}
    magg = {}
    for s, e, etype, payload, stream in req:
        if etype != 'M':
            continue
        k, nbytes = payload
        d = magg.setdefault(k, [0, 0, 0.0])
        d[0] += 1
        d[1] += nbytes
        d[2] += (e - s)
    rep['memcpy'] = {k: dict(count=v[0], bytes=v[1], ms=round(v[2] / 1e6, 3), gbps=round(v[1] / (v[2] / 1e9) / 1e9, 2) if v[2] else None)
                     for k, v in magg.items()}
    # busy time over all streams (union of intervals), and gap total
    ivs = sorted((s, e) for s, e, *_ in req)
    busy, cs, ce = 0, None, None
    for s, e in ivs:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            busy += ce - cs
            cs, ce = s, e
    busy += ce - cs
    rep['busy_ms'] = round(busy / 1e6, 3)
    rep['gap_ms'] = round((w1 - w0 - busy) / 1e6, 3)
    if a.rounds:
        rep['per_round'] = dict(rounds=a.rounds,
                                window_ms_per_round=round((w1 - w0) / 1e6 / a.rounds, 3),
                                kernel_ms_per_round=round(rep['kernel_total_ms'] / a.rounds, 3),
                                busy_ms_per_round=round(busy / 1e6 / a.rounds, 3),
                                gap_ms_per_round=round((w1 - w0 - busy) / 1e6 / a.rounds, 3))
    # API calls (syncs) in the window
    if 'CUDA_API_TRACE' in tables(con):
        cols = [r[1] for r in con.execute('pragma table_info(CUDA_API_TRACE)')]
        print('CUDA_API_TRACE cols:', cols)
        sel = ('select start, end, s.value from CUDA_API_TRACE t join StringIds s on s.id = t.shortName '
               'where t.start between ? and ?') if 'shortName' in cols else None
        if sel:
            api = list(con.execute(sel, (w0, w1)))
            aagg = {}
            for s, e, name in api:
                d = aagg.setdefault(name, [0, 0.0])
                d[0] += 1
                d[1] += (e - s)
            rep['api'] = [dict(name=n, count=v[0], ms=round(v[1] / 1e6, 3))
                          for n, v in sorted(aagg.items(), key=lambda kv: -kv[1][1])[:25]]
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main()
