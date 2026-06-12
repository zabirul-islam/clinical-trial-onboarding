#!/usr/bin/env python3
"""
Phase 0 sync gate — emit a checksum manifest (path,size,sha256) for given roots.

Run identically on local Mac and DGX server, then diff the two CSVs to find
local-only / server-only / hash-mismatch files before any phase-5 work.

Usage:
  python scripts_phase5/sync_manifest.py --out /tmp/manifest_local.csv
  python scripts_phase5/sync_manifest.py --roots outputs --out /tmp/m_outputs.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

DEFAULT_ROOTS = [
    "configs", "data", "indices", "outputs",
    "scripts", "scripts_phase3", "scripts_phase4", "scripts_phase5", "src",
]
EXCLUDE_DIRS = {"__pycache__", ".git", "logs", ".ipynb_checkpoints", "node_modules"}
EXCLUDE_SUFFIXES = {".log", ".pyc", ".DS_Store"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(root: Path, repo: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix in EXCLUDE_SUFFIXES or p.name in EXCLUDE_SUFFIXES:
            continue
        yield p.relative_to(repo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n = 0
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size", "sha256"])
        for root_name in args.roots:
            root = args.repo / root_name
            if not root.exists():
                print(f"[skip] missing root: {root_name}", file=sys.stderr)
                continue
            for rel in walk(root, args.repo):
                fp = args.repo / rel
                w.writerow([str(rel), fp.stat().st_size, sha256_of(fp)])
                n += 1
    print(f"[done] {n} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
