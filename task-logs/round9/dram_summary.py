#!/usr/bin/env python3
"""Match complete non-multiplexed DF counter intervals to accepted requests."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics
from typing import TypedDict


class Sample(TypedDict):
    config: str
    prompt: str
    epoch_end: float
    gb_s: float
    channel_gb_s: dict[str, float]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('counters', type=Path)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('requests', type=Path)
    args = parser.parse_args()
    meta = json.loads(args.metadata.read_text())
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    requests = [r for r in requests if r['kind'] == 'request' and r['accepted']]
    intervals = collections.defaultdict(dict)
    with args.counters.open() as stream:
        for row in csv.reader(stream):
            if len(row) < 6 or row[0].startswith('#'):
                continue
            match = re.fullmatch(r'amd_df/dram_channel_data_controller_(\d+)/', row[3])
            if match and row[1].isdigit():
                intervals[float(row[0])][int(match[1])] = (int(row[1]), float(row[5]))
    samples: list[Sample] = []
    previous = 0.0
    for end, channels in sorted(intervals.items()):
        begin, previous = previous, end
        if set(channels) != set(meta['channels']) or any(pct < 99.9 for _, pct in channels.values()):
            continue
        epoch_begin, epoch_end = meta['start_epoch'] + begin, meta['start_epoch'] + end
        for request in requests:
            if epoch_begin >= request['epoch'] - request['wall_s'] and epoch_end <= request['epoch']:
                by_channel = {str(ch): count * meta['bytes_per_event'] / (end-begin) / 1e9
                              for ch, (count, _) in channels.items()}
                samples.append(dict(config=request['config'], prompt=request['prompt'], epoch_end=epoch_end,
                                    gb_s=sum(by_channel.values()), channel_gb_s=by_channel))
                break
    summary = []
    for config in sorted({r['config'] for r in samples}):
        selected = [r for r in samples if r['config'] == config]
        values = [r['gb_s'] for r in selected]
        summary.append(dict(config=config, samples=len(selected), mean_gb_s=statistics.mean(values),
                            min_gb_s=min(values), max_gb_s=max(values)))
    print(json.dumps(dict(summary=summary, samples=samples,
        caveat='System-wide traffic during complete request intervals, not isolated CPU peak bandwidth; perf startup and rounded request timestamps introduce subsecond alignment uncertainty.'), indent=2))


if __name__ == '__main__':
    main()
