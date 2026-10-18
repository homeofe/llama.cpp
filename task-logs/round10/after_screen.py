#!/usr/bin/env python3
"""Select two screened candidates by recorded rules and confirm them in rotating rounds."""
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

from summarize import summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round10'
DATA = Path('/opt/llm-gateway/data/moe-quant/qwen38')


def archive(source, destination):
    destination.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.suffix in ['.json', '.jsonl', '.log', '.py']:
            shutil.copy2(path, destination / path.name)


def main():
    screen = DATA / 'round10-screen'
    while True:
        rows = [json.loads(l) for l in (screen / 'runs.jsonl').read_text().splitlines()]
        if rows[-1]['kind'] == 'all_complete':
            break
        if rows[-1]['kind'] == 'stop' or time.time() > 1792310700:  # 08:05 UTC
            raise TimeoutError('screen did not complete; no confirmation selection made')
        time.sleep(5)
    result = summarize(screen)
    assert result['complete']
    assert len(result['summary']) == 16 and all(r['requests'] == 4 for r in result['summary'])
    (OUT / 'screen-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    archive(screen, OUT / 'screen-results')
    grouped = {}
    for row in result['summary']:
        grouped.setdefault(row['config'], {})[row['category']] = row['tok_s']
    control = grouped.pop('ling-cache-mtp3')
    eligible = [n for n in grouped if grouped[n]['code'] > control['code'] and grouped[n]['english'] >= control['english']]
    ranked = sorted(eligible, key=lambda n: min(grouped[n][c]/control[c] for c in ['code', 'english']), reverse=True)
    for name in sorted(grouped, key=lambda n: grouped[n]['code'], reverse=True):
        if name not in ranked:
            ranked.append(name)
    balanced, code = ranked[:2]
    utc = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    selection = dict(rule='Prefer candidates beating code and tying/beating English; rank by minimum category/control ratio. Fill any remaining slot by code speed.',
                     control='ling-cache-mtp3', balanced=balanced, second=code, selected=[balanced, code],
                     screening_only=True, utc=utc)
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection), flush=True)
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(utc + ' Round 10: Screening abgeschlossen; drei Bestaetigungsrunden fuer ' + balanced + ' und ' + code + ' gegen MTP3.\n')
    command = ['python3', '-u', str(OUT / 'bench.py'), str(DATA / 'round10-confirm'), '09:00', '--rounds', '3',
               '--only', 'ling-cache-mtp3', balanced, code]
    process = None
    try:
        with (OUT / 'confirm.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        code_exit = process.wait()
        if code_exit:
            raise RuntimeError('confirmation process failed')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=65)
    result = summarize(DATA / 'round10-confirm')
    assert result['complete'] and len(result['summary']) == 6
    assert all(r['requests'] == 12 and r['measured_lengths'] == 12 for r in result['summary'])
    (OUT / 'confirm-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    archive(DATA / 'round10-confirm', OUT / 'confirm-results')
    (OUT / 'confirmation-complete.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), selected=selection['selected']))+'\n')
    print('confirmation completed', flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
