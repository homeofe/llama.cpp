import ctypes,time
n=ctypes.CDLL("libnvToolsExt.so.1")
n.nvtxRangePushA(b"round10")
end=time.monotonic()+0.2
x=0
while time.monotonic()<end: x+=1
n.nvtxRangePop()
