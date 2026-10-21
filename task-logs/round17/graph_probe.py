#!/usr/bin/env python3
"""Record graph-property changes on real warmed requests; not a benchmark."""
import argparse, json, pathlib, signal, sys, time
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'round16'))
from run_variants import GraphRun, configs, bench

ap=argparse.ArgumentParser()
ap.add_argument('--out',required=True)
ap.add_argument('--runtime',required=True,type=pathlib.Path)
ap.add_argument('--trace',action='store_true')
ap.add_argument('--capacity',type=int,default=1)
ap.add_argument('--minutes',type=int,default=10)
a=ap.parse_args()
out=HERE/a.out;out.mkdir(exist_ok=False)
run=GraphRun(out,time.time()+60*a.minutes)
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
cfg=configs()['strata-reference'];cfg['cmd'][0]=str(a.runtime.resolve()/'llama-server')
cfg['env']['GGML_CUDA_GRAPH_VARIANTS']=str(a.capacity)
if a.trace:cfg['env']['GGML_CUDA_GRAPH_TRACE']='1'
(out/'config.json').write_text(json.dumps({'config':cfg,'benchmark_valid':False},indent=2)+'\n')
try:
 run.quiet();run.start('strata-reference',cfg)
 run.event(kind='warmup',benchmark_valid=False,**run.request('Say hello in five words.','warmup',16))
 for tag,prompt in [('code-novel',bench.PROMPTS[0][1]),('code-repeat',bench.PROMPTS[0][1]),('english',bench.PROMPTS[4][1])]:
  run.event(kind='request',tag=tag,benchmark_valid=False,**run.request(prompt,tag,256))
finally:
 run.stop();run.event(kind='released',gpu=run.gpu_apps())
