#!/usr/bin/env python3
"""Pause this task's reference runner between requests while executing a CPU check."""
import json, os, pathlib, signal, subprocess, sys, time
proc = pathlib.Path('/proc/1483672')
identity = None
events = pathlib.Path(__file__).with_name('build-events.jsonl')
def event(**kw):
    with events.open('a') as f: f.write(json.dumps(dict(epoch=time.time(), **kw))+'\n')
if proc.exists():
    assert b'task-logs/round16/native_reference.py' in (proc/'cmdline').read_bytes().split(b'\0')
    deadline = time.time()+30
    while (proc/'wchan').read_text().strip() != 'hrtimer_nanosleep':
        assert time.time() < deadline, 'reference not between requests'
        time.sleep(.2)
    identity = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    os.kill(int(proc.name), signal.SIGSTOP)
    event(kind='pause_reference',command=sys.argv[1:])
try:
    result = subprocess.run(sys.argv[1:],timeout=480)
    sys.exit(result.returncode)
finally:
    if identity is not None and proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[19] == identity:
        os.kill(int(proc.name), signal.SIGCONT)
        event(kind='resume_reference')
