"""Stand-in for rclone in tests, between local folders.

    python fake_rclone.py copy  SRC DST [--include PATTERN ...]
    python fake_rclone.py check SRC DST --one-way [--include PATTERN ...]
    python fake_rclone.py lsf   DIR [--include PATTERN ...]

FAKE_RCLONE_FAIL=copy|check|lsf makes that command fail (to test error handling).
"""
import fnmatch
import os
import shutil
import sys
from pathlib import Path

cmd, *rest = sys.argv[1:]
patterns = [rest[i + 1] for i, a in enumerate(rest) if a == "--include"] or ["*"]
paths = [a for i, a in enumerate(rest) if not a.startswith("--") and (i == 0 or rest[i - 1] != "--include")]
if os.environ.get("FAKE_RCLONE_FAIL") == cmd:
    print(f"fake rclone: {cmd} failed", file=sys.stderr)
    sys.exit(1)


def matching(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir())
            if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns)]


if cmd == "lsf":
    for p in matching(paths[0]):
        print(p.name)
elif cmd == "copy":
    src, dst = paths[:2]
    Path(dst).mkdir(parents=True, exist_ok=True)
    for p in matching(src):
        shutil.copy2(p, Path(dst) / p.name)
elif cmd == "check":
    src, dst = paths[:2]
    bad = [p.name for p in matching(src)
           if not (Path(dst) / p.name).exists() or (Path(dst) / p.name).read_bytes() != p.read_bytes()]
    if bad:
        print(f"fake rclone: differences: {bad}", file=sys.stderr)
        sys.exit(1)
else:
    sys.exit(f"fake rclone: unsupported command {cmd}")
