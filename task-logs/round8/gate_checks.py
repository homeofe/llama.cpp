#!/usr/bin/env python3
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
logs = root / 'task-logs/round8/gate-tests'
logs.mkdir(exist_ok=True)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

stage = module('stage_b', '/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py')
step = module('gpu_step', root / 'task-logs/round8/gpu_step.py')
run = stage.Run(logs, float('inf'), True)
run.start = Mock(side_effect=RuntimeError('start failed'))
run.stop = Mock()
try:
    run.one_pass('q-warm-draft', stage.CONFIGS['q-warm-draft'], [], 1)
except RuntimeError:
    pass
run.stop.assert_called_once()

run = stage.Run(logs, float('inf'), True)
run.start = Mock()
run.stop = Mock()
run.quiet = Mock(return_value=(0,0.1))
run.check_gpu = Mock()
run.event = Mock()
run.request = Mock(return_value={'decode_tok_s':42.0})
with patch.object(stage, 'PROMPTS', [('one','a'),('two','b')]), \
     patch.object(stage, 'busy_jiffies', return_value=0), patch.object(stage, 'proc_jiffies', return_value=0), \
     patch.object(stage.os, 'getloadavg', side_effect=[(0.1,0,0),(0.1,0,0),(0.1,0,0),(3.1,0,0)]), \
     patch.object(stage.subprocess, 'check_output', return_value='100'):
    try:
        run.one_pass('candidate', {}, ['one','two'], 1)
        raise AssertionError('a contaminated pass should be repeated in full')
    except stage.Requeue:
        pass
    assert not run.done
    assert all(c.kwargs['kind'] == 'request_pending' for c in run.event.call_args_list)
run.stop.assert_called_once()

run = stage.Run(logs, float('inf'), True)
run.held = True
run.renew = Mock()
with patch.object(stage.os, 'getloadavg', return_value=(3.1, 0, 0)):
    try:
        run.quiet()
        raise AssertionError('busy load should release the lease')
    except stage.Requeue:
        pass

run = stage.Run(logs, float('inf'), True)
run.gpu_apps = Mock(side_effect=[{}, {}, {12345: 1024}, {12345: 1024}])
child = Mock()
child.poll.return_value = None
response = Mock()
response.__enter__ = Mock(return_value=Mock(status=200))
response.__exit__ = Mock(return_value=False)
with patch.object(stage, 'gw') as gateway, patch.object(stage.subprocess, 'check_output', return_value=''), \
     patch.object(stage.subprocess, 'Popen', return_value=child) as popen, \
     patch.object(stage.urllib.request, 'urlopen', return_value=response), \
     patch.dict(stage.os.environ, {'GGML_MOE_HYBRID_BYTE_VALUE':'1', 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH':'8'}):
    run.start('q-warm-draft', stage.CONFIGS['q-warm-draft'])
    assert run.gpu_pid == 12345
    assert 'GGML_MOE_HYBRID_BYTE_VALUE' not in popen.call_args.kwargs['env']
    assert 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH' not in popen.call_args.kwargs['env']
    assert gateway.call_args_list[0].args[:2] == ('POST','/api/v1/gpu/free')
    assert gateway.call_args_list[1].args[2] == {'reason':'codex: round 8','ttlS':900}

with patch.object(step, 'apps', side_effect=RuntimeError('unavailable')), patch.object(step, 'gateway') as gateway, \
     patch.object(sys, 'argv', ['gpu_step.py',str(logs/'unavailable'),'--','unused']):
    assert step.main() == 77
    gateway.assert_not_called()

child = Mock()
child.poll.return_value = 0
child.returncode = 0
with patch.object(step, 'apps', return_value={}), patch.object(step, 'gateway') as gateway, \
     patch.object(step.subprocess, 'Popen', return_value=child), \
     patch.object(step.os, 'getloadavg', side_effect=[(0.1,0,0),(0.1,0,0),(3.1,0,0)]*3), \
     patch.object(sys, 'argv', ['gpu_step.py',str(logs/'busy-end'),'--','unused']):
    assert step.main() == 75
    assert sum(c.args == ('DELETE','hold') for c in gateway.call_args_list) == 3
    assert not (logs/'busy-end/accepted.txt').exists()

child = Mock(pid=98765)
child.poll.return_value = None
child.wait.side_effect = [subprocess.TimeoutExpired('unused',5),0]
with patch.object(step, 'apps', side_effect=[{},{},{1:1024,2:1024}]), \
     patch.object(step, 'gateway') as gateway, patch.object(step.subprocess, 'Popen', return_value=child), \
     patch.object(step.os, 'getloadavg', return_value=(0.1,0,0)), patch.object(step.os, 'killpg') as kill, \
     patch.object(sys, 'argv', ['gpu_step.py',str(logs/'foreign'),'--','unused']):
    try:
        step.main()
        raise AssertionError('foreign GPU process should stop the child')
    except RuntimeError:
        pass
    kill.assert_called_once_with(98765, step.signal.SIGTERM)
    assert gateway.call_args.args == ('DELETE','hold')
print('Gate checks passed: no lease without GPU; cleanup after start failure; both load boundaries; foreign VRAM abort; inherited controls isolated')
