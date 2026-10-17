#!/usr/bin/env python3
import difflib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
logs=root/'task-logs/round8'
external=[]
for before, after, label in [
    (logs/'stage_b.before.py',Path('/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py'),'moe-quant/qwen38/stage_b.py'),
    (logs/'record-traces.before.sh',Path('/opt/llm-gateway/data/moe-quant/record-traces.sh'),'moe-quant/record-traces.sh'),
]:
    external.extend(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile='a/'+label,tofile='b/'+label))
(logs/'measurement.patch').write_text(''.join(external))
files=['NOTES-round8.md']
for path in sorted(logs.iterdir()):
    if path.is_file() and (path.suffix in ('.cpp','.sh','.py','.md') or path.name in ('preflight.json','final-device-check.json','measurement.patch')):
        if '.before.' not in path.name and path.name not in ('rewrite_replay.py','edit_stage_b.py'):
            files.append(str(path.relative_to(root)))
base=json.loads((logs/'preflight.json').read_text())['base']
patch=subprocess.check_output(['git','diff','--binary',base],cwd=root)
for file in files:
    if subprocess.run(['git','ls-files','--error-unmatch',file],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        result=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',file],cwd=root,capture_output=True)
        assert result.returncode in (0,1),result.stderr.decode()
        patch+=result.stdout
out=logs/(sys.argv[1] if len(sys.argv)>1 else 'round8.patch')
out.write_bytes(patch)
print(out, len(patch), 'bytes')
