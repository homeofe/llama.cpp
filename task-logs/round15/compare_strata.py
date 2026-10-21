#!/usr/bin/env python3
"""Reproduce the trace-visibility audit and the draft-head size/time bound."""
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STRATA = Path('/opt/llm-gateway/data/runtimes/strata/0.1.41-local-1cb045cb')
sys.path.insert(0, str(ROOT / 'gguf-py'))
import gguf


def trace(path):
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    count, graph_nodes, ms = c.execute('SELECT count(*), count(graphNodeId), sum(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    calls = c.execute('SELECT s.value, count(*), sum(r.end-r.start)/1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE "%cudaGraph%" GROUP BY s.value').fetchall()
    return dict(path=str(path), kernel_count=count, graph_node_ids=graph_nodes, visible_kernel_ms=ms,
                graph_api_calls=[dict(name=n, count=k, host_ms=t) for n, k, t in calls])


def main():
    draft_path = Path('/opt/llm-gateway/models/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf')
    reader = gguf.GGUFReader(draft_path)
    tensors = {t.name: t for t in reader.tensors}
    head = tensors['output.weight']
    experts = [t for t in reader.tensors if '_exps.weight' in t.name]
    raw = Path('/opt/llm-gateway/data/strata-data/mtp/rt/draft_vocab.bin').read_bytes()
    ids = struct.unpack('<' + str(len(raw) // 4) + 'i', raw)
    n_vocab = int(head.shape[1])
    assert len(set(ids)) == len(ids) and min(ids) >= 0 and max(ids) < n_vocab
    assert head.tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert len(experts) == 3 and all(t.tensor_type == gguf.GGMLQuantizationType.Q8_0 for t in experts)
    old = trace(ROOT / 'task-logs/vibe-round1/strata-shim/strata-shim.sqlite')
    ours = trace(ROOT / 'task-logs/vibe-round14/trace.sqlite')
    c = sqlite3.connect(f'file:{ours["path"]}?mode=ro', uri=True)
    rows = c.execute('SELECT s.value, count(*), sum(k.end-k.start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.gridX=248320 AND k.gridY=1 AND k.gridZ=1 AND s.value LIKE "void mul_mat_vec_q<(ggml_type)8, (int)1,%" GROUP BY s.value').fetchall()
    assert len(rows) == 1 and rows[0][1] == 180
    rounds = 60
    head_ms = rows[0][2] / rounds
    q8_subset_bytes = head.n_bytes * len(ids) // n_vocab
    # The native target head is type 14, Q6_K: 210 bytes for 256 weights.
    q6_subset_bytes = len(ids) * int(head.shape[0]) // 256 * 210
    sources = [STRATA / 'src/core/verify.cpp', STRATA / 'include/strata/core/verify.hpp',
               STRATA / 'src/core/mtp.cpp', STRATA / 'tools/mtp_rt.py',
               ROOT / 'src/models/qwen4exp.cpp', ROOT / 'src/models/dflash.cpp',
               ROOT / 'ggml/src/ggml-backend.cpp']
    report = dict(
        versions=dict(llama_base='6c8bfda89', strata_measured='0.1.41-local-1cb045cb',
                      strata_upstream_head='9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4'),
        trace_audit=dict(strata_old=old, llama_round14=ours,
                         verdict='Old Strata capture omits graph nodes; its visible kernel sum is not total GPU work.'),
        draft=dict(path=str(draft_path), architecture=reader.fields['general.architecture'].contents(),
                   full_vocab=n_vocab, subset_vocab=len(ids), subset_ids_sha256=hashlib.sha256(raw).hexdigest(),
                   q8_head_bytes=head.n_bytes, q8_subset_bytes=q8_subset_bytes,
                   q8_subset_saving_mib=(head.n_bytes-q8_subset_bytes)/2**20,
                   strata_native_subset_bytes=q6_subset_bytes,
                   q8_vs_strata_head_saving_mib=(head.n_bytes-q6_subset_bytes)/2**20,
                   q8_experts_bytes=sum(t.n_bytes for t in experts),
                   strata_q2_experts_bytes=Path('/opt/llm-gateway/data/strata-data/mtp/rt/experts.bin').stat().st_size),
        draft_head_bound=dict(kernel=rows[0][0], launches=rows[0][1], rounds=rounds,
                              measured_graph_off_ms_per_round=head_ms,
                              row_fraction=len(ids)/n_vocab,
                              linear_subset_estimated_saving_ms=head_ms*(1-len(ids)/n_vocab),
                              limitations=['Graph-off trace; not a graph-on speedup measurement.',
                                           'Linear row scaling is an estimate, not a measured kernel time.',
                                           'Draft acceptance and target verify-window numerics can change.',
                                           'VRAM savings do not include extra mappings, graph storage or buffers.']),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT / 'strata-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(trace_audit=report['trace_audit'], draft=report['draft'], bound=report['draft_head_bound']), indent=2))


if __name__ == '__main__':
    main()
