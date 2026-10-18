import ctypes,subprocess,sys
child=subprocess.Popen([sys.executable,"-c","import time; print(\"ready\",flush=True); input(); end=time.monotonic()+0.2; x=0\nwhile time.monotonic()<end: x+=1"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
assert child.stdout is not None
assert child.stdout.readline().strip()=="ready"
n=ctypes.CDLL("libnvToolsExt.so.1")
n.nvtxRangePushA(b"round10")
child.communicate("go\n")
n.nvtxRangePop()
print("sampled_child_pid",child.pid)
assert child.returncode==0
