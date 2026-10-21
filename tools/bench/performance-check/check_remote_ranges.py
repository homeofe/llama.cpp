#!/usr/bin/env python3
"""Exercise the real range reader against a local HTTP server. No Hub access."""
import contextlib
import http.server
import io
import pathlib
import sys
import threading
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gguf_remote


@contextlib.contextmanager
def source(status=206, content_range='bytes 0-31/32', body=b'x' * 32, length=None):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header('Content-Range', content_range)
            self.send_header('Content-Length', str(len(body) if length is None else length))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/model.gguf'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class RangeChecks(unittest.TestCase):
    def test_exact_tensor_range(self):
        with source(content_range='bytes 7-10/32', body=b'data') as url:
            out = io.BytesIO()
            gguf_remote.copy_range(url, 7, 4, out)
            self.assertEqual(out.getvalue(), b'data')

    def test_tensor_range_must_not_end_early(self):
        with source() as url, self.assertRaises(ValueError):
            gguf_remote.copy_range(url, 0, 33, io.BytesIO())

    def test_valid_short_final_header_range(self):
        with source() as url:
            self.assertEqual(gguf_remote.Remote(url).read(4), b'xxxx')

    def test_reject_ignored_range(self):
        with source(status=200) as url, self.assertRaises(ValueError):
            gguf_remote.Remote(url).read(4)

    def test_reject_wrong_offset(self):
        with source(content_range='bytes 1-32/33') as url, self.assertRaises(ValueError):
            gguf_remote.Remote(url).read(4)

    def test_reject_short_nonfinal_range(self):
        with source(content_range='bytes 0-31/64') as url, self.assertRaises(ValueError):
            gguf_remote.Remote(url).read(4)

    def test_reject_length_mismatch(self):
        with source(body=b'x' * 31) as url, self.assertRaises(ValueError):
            gguf_remote.Remote(url).read(4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
