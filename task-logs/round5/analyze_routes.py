import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('prefix')
parser.add_argument('--budget-mib', type=int, action='append')
parser.add_argument('--reserve-mib', type=int, default=768)
args = parser.parse_args()
prefix = Path(args.prefix)
report = json.loads(prefix.with_suffix('.report.json').read_text())
assert report['trace']
all_layers = {item['layer']: item for item in report['layers']}
layers = {il: layer for il, layer in all_layers.items() if all(t['host'] for t in layer['tensors'])}
assert layers
assert len(report['memory_steady']) == 1, 'analysis currently supports one GPU'
alignment = report['memory_steady'][0]['alignment']
assert alignment > 0
assert args.reserve_mib >= 0
groups = defaultdict(list)
expert_bytes = {}
for il, layer in layers.items():
    layout = tuple((t['type'], tuple(t['ne']), t['expert_bytes']) for t in layer['tensors'])
    assert all(t['host'] for t in layer['tensors'])
    groups[layout].append(il)
    expert_bytes[il] = sum(t['expert_bytes'] for t in layer['tensors'])
train = Counter()
test = Counter()
rows = []
seen = set()
topic_sizes = {t['topic']: len(t['tokens']) for t in report['topics']}
with prefix.with_suffix('.routes.jsonl').open() as source:
    for line in source:
        row = json.loads(line)
        key = (row['topic'], row['token'], row['layer'])
        assert key not in seen
        seen.add(key)
        if row['layer'] not in layers:
            continue
        n_expert = layers[row['layer']]['tensors'][0]['ne'][2]
        assert all(0 <= e < n_expert for e in row['ids'])
        assert len(row['ids']) == len(set(row['ids']))
        destination = train if row['token'] < topic_sizes[row['topic']] // 2 else test
        destination.update((row['layer'], e) for e in row['ids'])
        if destination is test:
            rows.append(row)
assert len(seen) == sum(topic_sizes.values()) * len(all_layers)
total_host = sum(expert_bytes[il] * layer['tensors'][0]['ne'][2] for il, layer in layers.items())
free = min(d['free'] for d in report['memory_steady'])
maximum_mib = max(0, (free // 2**20 - args.reserve_mib) // 256 * 256)
budgets = sorted(set(args.budget_mib or [2048, 3072, maximum_mib]) - {0})
assert all(mib > 0 for mib in budgets)
result = {'train_tokens': {k: v // 2 for k, v in topic_sizes.items()},
          'test_tokens': {k: v - v // 2 for k, v in topic_sizes.items()},
          'memory_steady': report['memory_steady'], 'reserve_mib': args.reserve_mib,
          'scope': 'host-resident expert layers', 'alignment': alignment,
          'maximum_budget_mib': maximum_mib, 'results': []}
for mib in budgets:
    if mib > maximum_mib:
        continue
    resident = set()
    oracle = set()
    group_rows = []
    for layout, layer_ids in groups.items():
        size = sum(t[2] for t in layout)
        n_expert = layout[0][1][2]
        host_bytes = size * n_expert * len(layer_ids)
        budget = int(mib * 2**20 * host_bytes / total_host)
        def allocated(n):
            maps = len(layer_ids) * ((n_expert * 4 + alignment - 1) // alignment * alignment)
            return maps + sum((t[2] * (n + 1) + alignment - 1) // alignment * alignment for t in layout)
        slots = 0
        while slots < n_expert * len(layer_ids) and allocated(slots + 1) <= budget:
            slots += 1
        if slots < max(len(row['ids']) for row in rows if row['layer'] in layer_ids):
            slots = 0
        candidates = [(il, e) for il in layer_ids for e in range(n_expert)]
        resident.update(sorted(candidates, key=lambda key: (-train[key], key))[:slots])
        oracle.update(sorted(candidates, key=lambda key: (-test[key], key))[:slots])
        group_rows.append({'layers': layer_ids, 'slots': slots, 'expert_bytes': size,
                           'allocated_bytes': allocated(slots) if slots else 0})
    topic = defaultdict(Counter)
    hist = Counter()
    layer_hist = defaultdict(Counter)
    for row in rows:
        il = row['layer']
        hits = sum((il, e) in resident for e in row['ids'])
        oracle_hits = sum((il, e) in oracle for e in row['ids'])
        counts = topic[row['topic']]
        counts.update(hits=hits, selected=len(row['ids']), byte_hits=hits * expert_bytes[il],
                      byte_selected=len(row['ids']) * expert_bytes[il], oracle_hits=oracle_hits)
        hist[hits] += 1
        layer_hist[il][hits] += 1
    totals = sum(topic.values(), Counter())
    record = {'budget_mib': mib, 'slots': len(resident), 'groups': group_rows,
              'allocated_mib': sum(g['allocated_bytes'] for g in group_rows) / 2**20,
              'hit_rate': totals['hits'] / totals['selected'],
              'byte_hit_rate': totals['byte_hits'] / totals['byte_selected'],
              'oracle_hit_rate': totals['oracle_hits'] / totals['selected'],
              'per_topic': {name: dict(c, hit_rate=c['hits'] / c['selected'],
                                    byte_hit_rate=c['byte_hits'] / c['byte_selected']) for name, c in topic.items()},
              'hits_per_layer_histogram': dict(sorted(hist.items())),
              'per_layer_histogram': {il: dict(sorted(c.items())) for il, c in layer_hist.items()}}
    result['results'].append(record)
    profile_path = prefix.parent / f'{prefix.name}-{mib}mib.profile'
    with profile_path.open('w') as out:
        out.write('llama-moe-profile-v1\n')
        for il, e in sorted(resident):
            out.write(f'{il} {e} {train[il, e]}\n')
    print(mib, 'MiB, slots', len(resident), 'held-out hits', round(record['hit_rate'] * 100, 3),
          'byte hits', round(record['byte_hit_rate'] * 100, 3))
prefix.with_suffix('.profile-summary.json').write_text(json.dumps(result, indent=2) + '\n')
