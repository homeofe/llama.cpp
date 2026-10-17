#!/usr/bin/env python3
"""Use stage_b's foreign-process gates and add output hashes and lease attribution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path('/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py')
spec = importlib.util.spec_from_file_location('stage_b', SOURCE)
assert spec and spec.loader
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class Run(stage.Run):
    def renew(self, force=False):
        if self.held and (force or time.monotonic() - self.last_hold > 300):
            stage.gw('POST', '/api/v1/gpu/hold', {'reason': 'codex: round 9', 'ttlS': 900})
            self.last_hold = time.monotonic()

    def request(self, *args, **kwargs):
        original = stage.urllib.request.urlopen
        chunks = []

        class Capture:
            def __init__(self, response):
                self.response = response

            def __enter__(self):
                self.response.__enter__()
                return self

            def __exit__(self, *exc):
                return self.response.__exit__(*exc)

            def __iter__(self):
                for line in self.response:
                    chunks.append(line)
                    yield line

        stage.urllib.request.urlopen = lambda *a, **kw: Capture(original(*a, **kw))
        try:
            result = super().request(*args, **kwargs)
        finally:
            stage.urllib.request.urlopen = original
        output = []
        for raw in chunks:
            line = raw.decode().strip()
            if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                continue
            for choice in json.loads(line[5:]).get('choices') or []:
                delta = choice.get('delta') or {}
                output.append(delta.get('content') or delta.get('reasoning_content') or '')
        result['text_sha256'] = hashlib.sha256(''.join(output).encode()).hexdigest()
        return result


if __name__ == '__main__':
    for key in list(os.environ):
        if key.startswith('GGML_MOE_HYBRID') or key == 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH':
            del os.environ[key]
    stage.Run = Run
    extra = Path(__file__).with_name('extra-configs.json')
    if extra.exists():
        stage.CONFIGS.update(json.loads(extra.read_text()))
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'harness.py').write_bytes(SOURCE.read_bytes())
    (out / 'configs.json').write_text(json.dumps(stage.CONFIGS, indent=2) + '\n')
    stage.main()
