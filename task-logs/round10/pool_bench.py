#!/usr/bin/env python3
"""Use the same stage-B pass gates for the original and persistent native pools."""
import json
import os
from pathlib import Path
import sys

import bench


class Run(bench.Run):
    def start(self, name, config):
        super().start(name, config)
        assert self.child is not None
        libraries = sorted({line.split()[-1] for line in Path(f'/proc/{self.child.pid}/maps').read_text().splitlines()
                            if '/libggml' in line or '/libllama' in line})
        directory = Path(config['cmd'][0]).parent
        native = config['env'].get('LD_PRELOAD')
        assert all(Path(path).parent == directory or path == native for path in libraries)
        cpu = [path for path in libraries if Path(path).name.startswith('libggml-cpu.so')]
        assert len(cpu) == 1 and (cpu[0] == native if native else Path(cpu[0]).parent == directory)
        (self.out / f'{name}-{self.child.pid}-libraries.json').write_text(json.dumps(dict(pid=self.child.pid,
            config=name, libraries=libraries, cpu_preload=native), indent=2) + '\n')


bench.configure()
bench.stage.Run = Run
config_path = Path(os.environ.get('R10_POOL_CONFIGS', str(Path(__file__).with_name('pool-configs.json'))))
bench.stage.CONFIGS.update(json.loads(config_path.read_text()))
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'configs.json').write_text(json.dumps(bench.stage.CONFIGS, indent=2) + '\n')
bench.stage.main()
