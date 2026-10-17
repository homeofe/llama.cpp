#!/usr/bin/env python3
"""Download the pinned public Ling draft and convert through existing support."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round9'
DEST = Path('/opt/llm-gateway/models/Ling-3.0-flash-GGUF/dspark')
REVISION = 'd60b0e1291dacb256194c6d48154398730cf8986'


def main():
    subprocess.run(['python3', '-S', 'task-logs/round9/extend_slots.py'], cwd=ROOT, check=True)
    DEST.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'HF_HOME': str(DEST / 'hf-cache'), 'HF_XET_CACHE': str(DEST / 'xet-cache'),
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    commands = [
        ['hf', 'download', 'inclusionAI/Ling-3.0-flash-dspark', '--revision', REVISION,
         '--include', '*.safetensors', '--include', '*.json', '--local-dir', str(DEST / 'hf')],
        ['hf', 'download', 'inclusionAI/Ling-3.0-flash', '--include', '*.json',
         '--include', '*.model', '--include', 'merges.txt', '--local-dir', str(DEST / 'target-metadata')],
        ['/opt/llm-gateway/data/venvs/convert/bin/python', '-B', 'convert_hf_to_gguf.py', str(DEST / 'hf'),
         '--target-model-dir', str(DEST / 'target-metadata'), '--outtype', 'q8_0',
         '--outfile', str(DEST / 'Ling-3.0-flash-DSpark-Q8_0.gguf')],
    ]
    (OUT / 'dspark-preparation.json').write_text(json.dumps(dict(commands=commands,
        draft_revision=REVISION, draft_quant='q8_0', production_code_changed=False), indent=2) + '\n')
    for i, cmd in enumerate(commands):
        with (OUT / f'dspark-prepare-{i}.log').open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
        print('step', i, 'complete', flush=True)
    with Path('/opt/llm-gateway/data/moe-quant/NOTE.log').open('a') as stream:
        stream.write(time.strftime('%H:%M:%S') + ' Runde 9: Ling DSpark mit vorhandener Konvertierung als Q8_0 exportiert; Laufzeitmessung folgt.\n')


if __name__ == '__main__':
    main()
