#!/usr/bin/env python3
"""Read existing server timestamps and speculative counters without a new build."""
import argparse
import json
from pathlib import Path
import re
from typing import Any


STAMP = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+) [A-Z] (.*)$')
STATS = re.compile(r'#calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+).*#gen tokens =\s*(\d+), #acc tokens =\s*(\d+).*dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms')


def parse(directory):
    epoch = json.loads((directory / 'logger-epoch.json').read_text())['epoch_us']
    entries = []
    stats = []
    decode_times = []
    for line in (directory / 'q-warm-draft.server.log').read_text(errors='replace').splitlines():
        match = STAMP.match(line)
        if not match:
            continue
        minute, second, milli, micro = map(int, match.groups()[:4])
        timestamp = epoch + ((minute * 60 + second) * 1000 + milli) * 1000 + micro
        message = match[5]
        entries.append(dict(epoch_us=timestamp, message=message))
        if 'prompt eval time' not in message:
            timing = re.search(r'eval time =\s*([\d.]+) ms /\s*(\d+) tokens', message)
            if timing:
                decode_times.append(dict(epoch_us=timestamp, ms=float(timing[1]), tokens=int(timing[2])))
        stat = STATS.search(message)
        if stat:
            stats.append(dict(epoch_us=timestamp,
                **dict(zip(['begin_calls', 'draft_calls', 'accept_calls', 'draft_tokens', 'accepted_tokens'], map(int, stat.groups()[:5]))),
                **dict(zip(['begin_ms', 'draft_ms', 'accept_ms'], map(float, stat.groups()[5:])))))
    rounds = []
    current: dict[str, Any] | None = None
    for event in entries:
        message, timestamp = event['message'], event['epoch_us']
        if 'posting NEXT_RESPONSE' in message:
            if current is not None:
                current['next_round_us'] = timestamp
                rounds.append(current)
            current = dict(start_us=timestamp, candidate_us={}, events=[])
        if current is None:
            continue
        if any(s in message for s in ['draft candidate', 'called impl', 'n_batch (effective)', 'add accepted tokens',
                                      'restoring speculative checkpoint', 'stop processing:', 'draft acceptance']):
            current['events'].append(event)
        candidate = re.search(r'draft candidate\s+0, pos\s+(\d+):', message)
        if candidate:
            current['candidate_us'][int(candidate[1])] = timestamp
        if 'called impl draft-mtp' in message:
            current['draft_end_us'] = timestamp
        if 'n_batch (effective)' in message:
            current['verify_start_us'] = timestamp
            batch = re.search(r'n_batch \(effective\) = (\d+)', message)
            if batch:
                current['batch_limit'] = int(batch[1])
        drafted = re.search(r'generate_draft:.*#draft=(\d+)', message)
        if drafted:
            current['drafted_tokens'] = int(drafted[1])
            current['verify_tokens'] = 1 + int(drafted[1])  # One slot: anchor plus drafts.
        if 'done yielding to queue' in message and 'verify_start_us' in current and 'accepted_us' not in current:
            current.setdefault('verify_yield_ends_us', []).append(timestamp)
        if 'add accepted tokens:' in message:
            current['accepted_us'] = timestamp
        if 'stop processing:' in message:
            current['release_us'] = timestamp
    if current is not None:
        rounds.append(current)
    stats_delta = None
    if len(stats) >= 2:
        stats_delta = {key: stats[-1][key] - stats[-2][key] for key in stats[-1] if key != 'epoch_us'}
    return dict(statistics=stats, request_statistics_delta=stats_delta, decode_times=decode_times, rounds=rounds,
        scope='Log timestamps mark existing call sites. Posting-to-candidate includes host setup and draft sampling. With exactly two post-verify queue-yield ends, the first ends target verification and the second ends MTP catch-up; accepted ends target sampling/acceptance. These are enclosing intervals, not isolated kernel timers. n_batch effective is capacity, not the actual verify width.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(parse(args.directory), indent=2))
