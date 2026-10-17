from pathlib import Path
p=Path('/opt/llm-gateway/data/moe-quant/qwen38/stage_b.py')
s=p.read_text()
needle="ENGLISH = ["
configs="""# Round 8 candidates. The policy and buffer changes are opt-in; existing controls stay intact.
for name, control, extra, env in [
    ('q-r8-policy', 'q-warm-draft', [], {'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
    ('ling-r8-policy', 'ling-cache-mtp3', [], {'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
    ('q-r8-ub32', 'q-warm-draft', ['--spec-draft-n-ubatch', '32'], {}),
    ('ling-r8-ub32', 'ling-cache-mtp3', ['--spec-draft-n-ubatch', '32'], {}),
    ('q-r8-slots', 'q-warm-draft', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '2944'}),
    ('ling-r8-slots', 'ling-cache-mtp3', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '5376'}),
    ('q-r8-combined', 'q-warm-draft', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '2944', 'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
    ('ling-r8-combined', 'ling-cache-mtp3', ['--spec-draft-n-ubatch', '32'], {'GGML_MOE_HYBRID_MIB': '5376', 'GGML_MOE_HYBRID_BYTE_VALUE': '1', 'GGML_MOE_HYBRID_DECAY': '64'}),
]:
    CONFIGS[name] = dict(cmd=CONFIGS[control]['cmd'] + extra, env={**CONFIGS[control]['env'], **env})

"""
assert needle in s
s=s.replace(needle,configs+needle,1)
s=s.replace('def __init__(self, out, end):','def __init__(self, out, end, load_gate=False):')
s=s.replace('self.out, self.end = out, end','self.out, self.end = out, end\n        self.load_gate = load_gate\n        self.gpu_pid = None',1)
s=s.replace("{'reason': 'claude: strata-parity stage A (llama.cpp)', 'ttlS': 900}","{'reason': 'codex: round 8' if self.load_gate else 'claude: strata-parity stage A (llama.cpp)', 'ttlS': 900}")
s=s.replace("            b0 = busy_jiffies(); time.sleep(5)\n            cores = (busy_jiffies() - b0) / HZ / 5\n            if cores < 1.0:\n                return round(time.time() - t0), round(cores, 2)","""            if self.load_gate:
                cores = os.getloadavg()[0]
                if cores < 3:
                    if self.child is not None:
                        self.check_gpu()
                    return round(time.time() - t0), round(cores, 2)
            else:
                b0 = busy_jiffies(); time.sleep(5)
                cores = (busy_jiffies() - b0) / HZ / 5
                if cores < 1.0:
                    return round(time.time() - t0), round(cores, 2)""")
s=s.replace('    def start(self, name, cfg):',"""    def gpu_apps(self):
        text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        return {int(row.split(',')[0]): int(row.split(',')[1]) for row in text.splitlines() if row.strip()}

    def check_gpu(self):
        foreign = {pid: mib for pid, mib in self.gpu_apps().items() if pid != self.gpu_pid and mib > 300}
        if foreign:
            raise Requeue('foreign GPU process: ' + str(foreign))

    def start(self, name, cfg):""")
s=s.replace("        gw('POST', '/api/v1/gpu/free', {})", "        if self.load_gate:\n            self.gpu_apps()\n        gw('POST', '/api/v1/gpu/free', {})",1)
s=s.replace("        log = open(self.out / f'{name}.server.log', 'a')", "        previous_gpu_pids = set(self.gpu_apps()) if self.load_gate else set()\n        log = open(self.out / f'{name}.server.log', 'a')",1)
s=s.replace("        self.child = subprocess.Popen(cfg['cmd'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,\n                                      env={**os.environ, **cfg.get('env', {})}, start_new_session=True)","""        env = dict(os.environ)
        if self.load_gate:
            for key in list(env):
                if key.startswith('GGML_MOE_HYBRID') or key == 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH':
                    del env[key]
        self.child = subprocess.Popen(cfg['cmd'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      env={**env, **cfg.get('env', {})}, start_new_session=True)
        log.close()""")
s=s.replace("            time.sleep(2)\n\n    def stop(self):","""            time.sleep(2)
        if self.load_gate:
            new_pids = set(self.gpu_apps()) - previous_gpu_pids
            if len(new_pids) != 1:
                raise RuntimeError('cannot identify the server GPU PID: ' + str(new_pids))
            self.gpu_pid = new_pids.pop()
            self.check_gpu()

    def stop(self):""")
s=s.replace('        self.child = None\n        if self.held:', '        self.child = None\n        self.gpu_pid = None\n        if self.held:')
s=s.replace("        self.start(name, cfg)\n        try:","        try:\n            self.start(name, cfg)")
s=s.replace("                    pids = engine_pids(name)","                    pids = [] if self.load_gate else engine_pids(name)")
s=s.replace("                    ok = foreign < 1.0 and res['decode_tok_s'] is not None", """                    l1 = os.getloadavg()[0]
                    if self.load_gate:
                        self.check_gpu()
                    ok = ((l0 < 3 and l1 < 3) if self.load_gate else foreign < 1.0) and res['decode_tok_s'] is not None""")
s=s.replace("pre_busy_cores=pre, load_start=l0, wall_s=round(wall, 2), foreign_cores=round(foreign, 2),", "pre_busy_cores=None if self.load_gate else pre, load_start=l0, load_end=l1, gate='load' if self.load_gate else 'foreign', wall_s=round(wall, 2), foreign_cores=None if self.load_gate else round(foreign, 2),")
s=s.replace("    a = ap.parse_args()", "    ap.add_argument('--load-gate', action='store_true', help='sandbox: require 1-minute load below 3 before and after each request')\n    a = ap.parse_args()")
s=s.replace('    run = Run(out, end)', '    run = Run(out, end, a.load_gate)')
p.write_text(s)
