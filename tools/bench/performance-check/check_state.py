#!/usr/bin/env python3
"""Prove persistence, reset and disabled-state I/O with the built C API."""
import ctypes
import json
import os
import pathlib
import subprocess
import sys
import tempfile


def library(path):
    lib = ctypes.CDLL(path)
    lib.llama_state_set.argtypes = [ctypes.c_char_p] * 3
    lib.llama_state_get.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.llama_state_erase.argtypes = [ctypes.c_char_p] * 2
    for name in ('llama_state_set', 'llama_state_get', 'llama_state_erase'):
        getattr(lib, name).restype = ctypes.c_bool
    return lib


if len(sys.argv) > 2:
    lib = library(sys.argv[2])
    mode = sys.argv[1]
    b = ctypes.create_string_buffer(256)
    if mode == 'write':
        for k, v in [(b'hot.0', b'7 2 5'), (b'tuned.l1', b'GATE=3'), (b'place.test.decided', b'cache')]:
            assert lib.llama_state_set(b'audit', k, v)
    elif mode == 'read':
        for k, v in [(b'hot.0', b'7 2 5'), (b'tuned.l1', b'GATE=3'), (b'place.test.decided', b'cache')]:
            assert lib.llama_state_get(b'audit', k, b, len(b)) and b.value == v
    else:
        assert not lib.llama_state_get(b'audit', b'hot.0', b, len(b))
        if mode == 'off':
            assert not lib.llama_state_set(b'audit', b'hot.0', b'999')
            assert not lib.llama_state_erase(b'audit', b'hot.')
    print('PASS', mode)
else:
    here = pathlib.Path(__file__).resolve().parent
    libpath = str(here.parents[2] / 'build-cpu/bin/libllama.so')
    with tempfile.TemporaryDirectory() as temp:
        env = dict(os.environ, XDG_CACHE_HOME=temp)
        env.pop('LLAMA_MOE_STATE', None)
        env.pop('LLAMA_MOE_CACHE_PROFILE', None)
        command = [sys.executable, str(pathlib.Path(__file__).resolve())]
        subprocess.run(command+['write',libpath], env=env, check=True)
        subprocess.run(command+['read',libpath], env=env, check=True)
        state = pathlib.Path(temp)/'llama.cpp/moe-state.ini'
        before = state.read_bytes(), state.stat().st_mtime_ns
        trace = here/'state-off.strace'
        subprocess.run(['strace','-qq','-f','-e','trace=%file','-o',str(trace)]+command+['off',libpath], env=dict(env,LLAMA_MOE_STATE='0'), check=True)
        assert 'moe-state.ini' not in trace.read_text()
        assert before == (state.read_bytes(), state.stat().st_mtime_ns)
        state.unlink()
        subprocess.run(command+['reset',libpath], env=env, check=True)
        print(json.dumps(dict(fresh_process_persistence=True,disabled_file_syscalls=0,disabled_file_unchanged=True,delete_resets=True)))
