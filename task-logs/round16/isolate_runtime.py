#!/usr/bin/env python3
"""Preserve the original runtime and replace only the experiment's libraries."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=['isolated', 'global', 'reference', 'reference-dense', 'reference-packed', 'reference-packed-v2'], default='isolated')
args = parser.parse_args()
old = here / 'e2e/original'
new = here / 'e2e' / args.name
source = here.parents[1] / 'build-cuda/bin'
shutil.copytree(old, new, symlinks=True)
changed = {'libllama.so.0.7.0'}
if args.name.startswith('reference'):
    changed.add('libllama-common.so.0.7.0')
for name in changed:
    shutil.copy2(source / name, new / name)
manifest = {}
for file in old.iterdir():
    other = new / file.name
    if file.is_symlink():
        assert file.readlink() == other.readlink()
    elif file.is_file():
        a = hashlib.sha256(file.read_bytes()).hexdigest()
        b = hashlib.sha256(other.read_bytes()).hexdigest()
        assert (a != b) == (file.name in changed), file.name
        manifest[file.name] = dict(original=a, candidate=b, identical=a == b)
(here / (args.name+'-binaries.json')).write_text(json.dumps(manifest, indent=2)+'\n')
print('ISOLATED_RUNTIME_OK:', ', '.join(sorted(changed)))
