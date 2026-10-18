#!/usr/bin/env python3
"""Use the same stage-B pass gates for the original and persistent native pools."""
import json
from pathlib import Path
import sys

import bench

bench.configure()
bench.stage.CONFIGS.update(json.loads(Path(__file__).with_name('pool-configs.json').read_text()))
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'configs.json').write_text(json.dumps(bench.stage.CONFIGS, indent=2) + '\n')
bench.stage.main()
