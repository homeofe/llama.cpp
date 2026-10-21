#!/usr/bin/env python3
"""Preserve the original runtime and replace only libllama in a separate copy."""
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
old = here / 'e2e/original'
new = here / 'e2e/isolated'
source = here.parents[1] / 'build-cuda/bin'
shutil.copytree(old, new, symlinks=True)
changed = 'libllama.so.0.7.0'
shutil.copy2(source / changed, new / changed)
manifest = {}
for file in old.iterdir():
    other = new / file.name
    if file.is_symlink():
        assert file.readlink() == other.readlink()
    elif file.is_file():
        a = hashlib.sha256(file.read_bytes()).hexdigest()
        b = hashlib.sha256(other.read_bytes()).hexdigest()
        assert (a != b) == (file.name == changed), file.name
        manifest[file.name] = dict(original=a, candidate=b, identical=a == b)
(here / 'isolated-binaries.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('ISOLATED_RUNTIME_OK: only libllama differs')
