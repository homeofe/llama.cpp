#!/usr/bin/env python3
"""Prepare model settings from measured configurations without touching live settings."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'task-logs/round11'
BINARY = '__CLAUDE_LLAMA_SERVER_BINARY__'


def entry(cfg, model_id, label, sampling):
    flags = cfg['cmd'][1:]
    model = Path(flags[flags.index('-m') + 1])
    key = re.sub(r'-\d{5}-of-\d{5}\.gguf$', '', str(model.relative_to('/opt/llm-gateway/models')))
    if key.endswith('.gguf'):
        key = key[:-5]
    args = []
    i = 0
    while i < len(flags):
        if flags[i] in ['-m', '--host', '--port', '--alias', '-lv']:
            i += 2
        else:
            args.append(flags[i])
            i += 1
    env = {k: v for k, v in cfg['env'].items() if k not in ['LD_LIBRARY_PATH', 'CUDA_CACHE_PATH']}
    value = dict(id=model_id, label=label, runtime='llama.cpp', binary=BINARY, args=args, env=env,
        sampling=sampling, reasoning='on', capabilities=dict(vision=False, tools=True, reasoning=True,
            context=int(flags[flags.index('-c') + 1])))
    return key, value


def main():
    configs = json.loads((OUT / 'configs.json').read_text())
    decision = json.loads((OUT / 'adoption.json').read_text())
    fit = json.loads((OUT / 'context-fit.json').read_text())
    panels = json.loads((OUT / 'panels/summary.json').read_text())
    assert panels['complete'], 'complete panel evidence required'
    ling = configs['ling-32k']
    assert (ling['env']['GGML_MOE_HYBRID_ASYNC_UPLOAD'] == '1') == decision['ling']['adopted']
    key, value = entry(ling, 'ling-3.0-flash-iq4_xs', 'Ling 3.0 Flash IQ4_XS - DSpark n3 - 32K',
        dict(temperature=0.6, top_p=0.95, top_k=20))
    models = {key: value}
    assert not decision['qwen']['adopted'], 'Qwen did not pass this round\'s speed gate; re-review before preparing an entry'
    card = Path('/opt/llm-gateway/data/moe-quant/round10-metadata/target/README.md')
    candidate = dict(status='prepared-not-applied', models=models,
        binary_resolution=dict(placeholder=BINARY, owner='Claude chooses and installs the runtime',
            reference_runtime=ling['cmd'][0],
            requirement='Preserve the measured private-fork runtime, CPU and CUDA libraries; a stock llama.cpp binary is not equivalent.',
            library_path='The gateway sets LD_LIBRARY_PATH to the selected binary directory.'),
        sampling_provenance=dict(model='Ling-3.0-flash', revision='fcc60a52b34c69c6271d0ac8a8cb0815020b6a43',
            source='https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/fcc60a52b34c69c6271d0ac8a8cb0815020b6a43/README.md',
            local_path=str(card), sha256=hashlib.sha256(card.read_bytes()).hexdigest()),
        evidence=dict(context_fit=fit, hard=dict(right=panels['hard']['right'], total=panels['hard']['total']),
            agent=dict(right=panels['agent']['right'], total=panels['agent']['total']), adoption=decision),
        application='Only the models entries are settings values. Replace the binary placeholder before installation. No live settings, runtime, keepLoaded list or bench history was modified.')
    (OUT / 'gateway-candidate.json').write_text(json.dumps(candidate, indent=2) + '\n')
    print(json.dumps(dict(status=candidate['status'], model_keys=list(models)), indent=2))


if __name__ == '__main__':
    main()
