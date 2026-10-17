#!/usr/bin/env python3
"""Convert the round-4 routes to ubatch logs; reserve tokens 592+ for the base profile."""
import collections
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
out = pathlib.Path(__file__).parent
rows = collections.defaultdict(lambda: collections.defaultdict(dict))
for line in source.open():
    r = json.loads(line)
    rows[r['topic']][r['token']][r['layer']] = r['ids']
topics = ['de', 'en', 'code', 'agent']
counts = collections.Counter()
for topic in topics:
    for token, layers in rows[topic].items():
        if token >= 592:
            for layer, ids in layers.items():
                counts.update((layer, expert) for expert in ids)
with (out / 'base.profile').open('w') as f:
    f.write('llama-moe-profile-v1\n')
    for (layer, expert), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        f.write(f'{layer} {expert} {count}\n')


def write_log(name, segments):
    with (out / name).open('w') as f:
        ubatch = 0
        for topic, start, length in segments:
            for offset in range(0, length):
                if offset != 0 and (offset < 40 or (offset - 40) % 4):
                    continue
                size = 40 if offset == 0 else min(4, length - offset)
                tokens = list(range(start + offset, start + offset + size))
                layers = []
                for layer in sorted(rows[topic][tokens[0]]):
                    layers.append(dict(layer=layer, cache=offset >= 40, expert_bytes=2050000,
                                       rows=list(range(size)), ids=[rows[topic][t][layer] for t in tokens]))
                record = dict(version=1, context=0, ubatch=ubatch, n_tokens=size,
                              positions=list(range(offset, offset + size)), seq_ids=[[0]] * size,
                              prefill=offset < 40, prefill_end=offset == 0, layers=layers)
                f.write(json.dumps(record, separators=(',', ':')) + '\n')
                ubatch += 1


write_log('long.jsonl', [(topic, 0, 1024) for topic in topics])
write_log('short.jsonl', [(topics[i % 4], (i // 4) * 296, 296) for i in range(8)])
