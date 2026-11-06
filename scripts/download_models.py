"""Download models from configs/models.yaml into $HF_HOME.

Usage: python scripts/download_models.py [group] [name ...]
  group defaults to "teachers"; names default to every model in the group.
"""
import os
import subprocess
import sys

import yaml

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(os.path.join(root, "configs", "models.yaml")))
group = sys.argv[1] if len(sys.argv) > 1 else "teachers"
names = sys.argv[2:] or list(cfg[group] or {})

failed = []
for name in names:
    m = cfg[group][name]
    cmd = ["hf", "download", m["id"]]
    for pat in m.get("exclude", []):
        cmd += ["--exclude", pat]
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        failed.append(name)

print("\nFAILED:", failed if failed else "none")
sys.exit(1 if failed else 0)
