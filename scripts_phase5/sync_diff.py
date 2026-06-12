#!/usr/bin/env python3
"""
Phase 0 sync gate — diff two sync_manifest.py CSVs (local vs server).

Usage:
  python scripts_phase5/sync_diff.py /tmp/manifest_local.csv /tmp/manifest_server.csv
"""
from __future__ import annotations

import sys

import pandas as pd


def main() -> int:
    local_csv, server_csv = sys.argv[1], sys.argv[2]
    lo = pd.read_csv(local_csv).set_index("path")
    sv = pd.read_csv(server_csv).set_index("path")

    only_local = sorted(set(lo.index) - set(sv.index))
    only_server = sorted(set(sv.index) - set(lo.index))
    common = sorted(set(lo.index) & set(sv.index))
    mismatch = [p for p in common if lo.loc[p, "sha256"] != sv.loc[p, "sha256"]]

    print(f"local files:  {len(lo)}")
    print(f"server files: {len(sv)}")
    print(f"common: {len(common)}  identical: {len(common) - len(mismatch)}")
    print(f"\n== HASH MISMATCH ({len(mismatch)}):")
    for p in mismatch:
        print(f"  {p}  local={lo.loc[p,'size']}B server={sv.loc[p,'size']}B")
    print(f"\n== LOCAL-ONLY ({len(only_local)}):")
    for p in only_local:
        print(f"  {p}")
    print(f"\n== SERVER-ONLY ({len(only_server)}):")
    for p in only_server:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
