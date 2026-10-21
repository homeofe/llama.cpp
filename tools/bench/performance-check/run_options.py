#!/usr/bin/env python3
import json, pathlib, subprocess, sys
here = pathlib.Path(__file__).resolve().parent
root = here.parents[2]
label = sys.argv[1]
cases = sys.argv[2:] or ['cache-zero', 'flags', 'stock-spec', 'dense-state', 'manual-state']
results = {}
for case in cases:
    cmd = [str(root/'build-cpu/bin/check-options'), case]
    if case.endswith('-state'):
        cmd += ['/opt/llm-gateway/models/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf', f'/tmp/llama-audit-{label}-{case}.ini']
    with (here/(label+'-'+case+'.txt')).open('w') as out:
        r = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=60)
    results[case] = r.returncode
(here/(label+'.json')).write_text(json.dumps(results,indent=2)+'\n')
print(results)
sys.exit(any(results.values()))
