"""Stand-in for rclone in tests: `copy` and `check --one-way` between local folders.

    python fake_rclone.py copy SRC DST --include PATTERN
    python fake_rclone.py check SRC DST --one-way --include PATTERN

FAKE_RCLONE_FAIL=copy|check makes that command fail (to test error handling).
"""
import fnmatch
import os
import shutil
import sys
from pathlib import Path

cmd, src, dst, *rest = sys.argv[1:]
pattern = rest[rest.index("--include") + 1] if "--include" in rest else "*"
if os.environ.get("FAKE_RCLONE_FAIL") == cmd:
    print(f"fake rclone: {cmd} failed", file=sys.stderr)
    sys.exit(1)

files = [p for p in Path(src).iterdir() if p.is_file() and fnmatch.fnmatch(p.name, pattern)]
if cmd == "copy":
    Path(dst).mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copy2(p, Path(dst) / p.name)
elif cmd == "check":
    bad = [p.name for p in files
           if not (Path(dst) / p.name).exists() or (Path(dst) / p.name).read_bytes() != p.read_bytes()]
    if bad:
        print(f"fake rclone: differences: {bad}", file=sys.stderr)
        sys.exit(1)
else:
    sys.exit(f"fake rclone: unsupported command {cmd}")
