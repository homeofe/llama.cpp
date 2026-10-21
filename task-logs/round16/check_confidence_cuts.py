#!/usr/bin/env python3
"""Compile the checker against two broken private header copies; production stays intact."""
from pathlib import Path
import json
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
header = (ROOT/'common/speculative-confidence.h').read_text()
cases = {
    'top10': header.replace('i < cur->size; ++i) {\n        sum +=', 'i < std::min<size_t>(cur->size, 10); ++i) {\n        sum +='),
    'tiny': header.replace('float(1.0/sum)', 'float(1.0/sum) + 1e-6f'),
    'restored': header,
}
assert cases['top10'] != header and cases['tiny'] != header
results = {}
with tempfile.TemporaryDirectory(prefix='confidence-cuts-', dir=HERE) as tmp:
    tmp = Path(tmp)
    for name,source in cases.items():
        (tmp/'speculative-confidence.h').write_text(source)
        exe = tmp/'check'
        runtime = HERE/'e2e/reference'
        subprocess.run(['g++','-std=c++17','-O2','-include','algorithm','-I'+str(tmp),'-I'+str(ROOT/'include'),
                        '-I'+str(ROOT/'ggml/include'), str(HERE/'check_confidence.cpp'), '-L'+str(runtime),
                        '-Wl,-rpath,'+str(runtime),'-lllama','-lggml-cpu','-lggml-cuda','-lggml-base','-o',str(exe)], check=True)
        result = subprocess.run([str(exe)],capture_output=True,text=True)
        (HERE/('confidence-cut-'+name+'.txt')).write_text(result.stdout+result.stderr)
        results[name] = result.returncode
        assert result.returncode == (0 if name=='restored' else 1), (name,result.returncode)
(HERE/'confidence-cuts.json').write_text(json.dumps(results,indent=2)+'\n')
print(results)
