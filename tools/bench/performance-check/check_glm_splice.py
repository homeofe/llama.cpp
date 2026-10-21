#!/usr/bin/env python3
"""Run the splicer on a tiny sharded-metadata fixture, without external downloads."""
import pathlib
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / 'gguf-py'))
sys.path.insert(0, str(HERE.parents[1]))
import gguf
import gguf_remote
from check_remote_ranges import source


class SpliceChecks(unittest.TestCase):
    def test_mtp_head_is_a_single_file(self):
        self.splice(False)

    def test_existing_nextn_layer_is_not_counted_twice(self):
        self.splice(True)

    def splice(self, already_mtp):
        with tempfile.TemporaryDirectory() as tmp:
            trunk, head = (pathlib.Path(tmp) / n for n in ('trunk.gguf', 'head.gguf'))
            w = gguf.GGUFWriter(trunk, 'glm5-next')
            w.add_block_count(3 if already_mtp else 2)
            if already_mtp:
                w.add_key_value('glm5-next.nextn_predict_layers', 1, gguf.GGUFValueType.UINT32)
            w.add_key_value('split.count', 4, gguf.GGUFValueType.UINT16)
            w.add_key_value('split.no', 0, gguf.GGUFValueType.UINT16)
            w.add_key_value('split.tensors.count', 100, gguf.GGUFValueType.INT32)
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file(); w.close()
            with patch.object(gguf_remote, 'header') as remote:
                with patch.object(sys, 'argv', ['glm_splice_mtp.py', str(trunk), str(head)]):
                    with self.assertRaisesRegex(ValueError, 'sharded trunks require --mtp-only'):
                        runpy.run_path(str(HERE.parents[1] / 'glm_splice_mtp.py'), run_name='__main__')
                remote.assert_not_called()
            payload = np.arange(32, dtype=np.float32).tobytes()
            with source(content_range='bytes 0-127/128', body=payload) as url:
                def header(_):
                    return ({'split.count': 1}, [dict(name='blk.2.attn_norm.weight', dims=[32], type=0, offset=0, size=128)], 0)
                original_copy = gguf_remote.copy_range
                def copy(_, start, size, out, **kw):
                    return original_copy(url, start, size, out, **kw)
                with patch.object(gguf_remote, 'header', header), patch.object(gguf_remote, 'copy_range', copy):
                    with patch.object(sys, 'argv', ['glm_splice_mtp.py', '--mtp-only', str(trunk), str(head)]):
                        runpy.run_path(str(HERE.parents[1] / 'glm_splice_mtp.py'), run_name='__main__')
            r = gguf.GGUFReader(head)
            self.assertFalse(any(k.startswith('split.') for k in r.fields), 'MTP head must not reference nonexistent shards')
            self.assertEqual(r.fields['glm5-next.block_count'].contents(), 3)
            self.assertEqual(r.fields['glm5-next.nextn_predict_layers'].contents(), 1)
            self.assertEqual(r.tensors[0].data.tobytes(), payload)


if __name__ == '__main__':
    unittest.main(verbosity=2)
