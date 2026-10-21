#!/usr/bin/env python3
"""Run the production MMVF unpack helper on every BF16 bit pattern on sm_75."""
import json, pathlib, subprocess, sys, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[3]
HERE=pathlib.Path(__file__).resolve().parent
s=(ROOT/'ggml/src/ggml-cuda/mmvf.cu').read_text()
a=s.index('template <typename T>\nstatic __device__ __forceinline__ void mmvf_v16_unpack')
b=s.index('\ntemplate <typename T, int ROWS>',a)
cut=sys.argv[1] if len(sys.argv)>1 else 'none'
helper=s[a:b]
if cut=='pair':helper=helper.replace('f[2*k] = t.x', 'f[2*k] = t.y')
if cut=='tiny':helper=helper.replace('f[2*k] = t.x', 'f[2*k] = t.x + 1e-6f')
assert cut in ('none','pair','tiny')
cu=ROOT/'build-cuda/check-bf16.cu'
cu.write_text('''#include "convert.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
'''+helper+'''
__global__ void check(float * out) {
    unsigned int i=(blockIdx.x*blockDim.x+threadIdx.x)*8;
    if (i>=65536) return;
    uint4 v=make_uint4(i|((i+1)<<16),(i+2)|((i+3)<<16),(i+4)|((i+5)<<16),(i+6)|((i+7)<<16));
    mmvf_v16_unpack<nv_bfloat16>(v,out+i);
}
int main() {
    float * d=nullptr; std::vector<float> out(65536);
    if (cudaMalloc(&d,65536*sizeof(float))!=cudaSuccess) return 2;
    check<<<64,128>>>(d);
    if (cudaGetLastError()!=cudaSuccess || cudaMemcpy(out.data(),d,65536*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess) return 3;
    cudaFree(d);
    int failures=0;
    for(unsigned i=0;i<65536;i++) {
        unsigned want=i<<16, got; float expected;
        std::memcpy(&expected,&want,4); std::memcpy(&got,&out[i],4);
        if (std::isnan(expected) ? !std::isnan(out[i]) : got!=want) failures++;
    }
    std::printf("BF16 patterns=65536 failures=%d (NaNs compared by class)\\n",failures);
    return failures!=0;
}
''')
cmd=['nvcc','-ccbin','/usr/bin/g++-12','-std=c++17','-O2','-arch=sm_75','-I'+str(ROOT/'ggml/include'),'-I'+str(ROOT/'ggml/src'),'-I'+str(ROOT/'ggml/src/ggml-cuda'),str(cu),'-o',str(cu.with_suffix(''))]
subprocess.run(cmd,check=True)
key=pathlib.Path('/opt/llm-gateway/data/api-key').read_text().strip()
def gw(method,path,body=None):
    req=urllib.request.Request('http://127.0.0.1:8790'+path,method=method,data=None if body is None else json.dumps(body).encode(),headers={'X-API-Key':key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r: return r.read()
gw('POST','/api/v1/gpu/hold',{'reason':'codex: BF16 conversion correctness','ttlS':120})
try:
    gw('POST','/api/v1/gpu/free',{})
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    result=subprocess.run([str(cu.with_suffix(''))])
    assert result.returncode == (0 if cut=='none' else 1), (cut,result.returncode)
    print('PASS control',cut,'exit',result.returncode)
finally:
    gw('DELETE','/api/v1/gpu/hold')
