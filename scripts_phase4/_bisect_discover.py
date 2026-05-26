"""Bisect discover_records() perf bug. Times stages A-G per file. Read-only."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
import semantic_leak_judge as M  # noqa: E402


def p99(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * 0.99))]


def bisect(bb_filter: str | None = None):
    seen: dict = {}
    grand_total = time.time()
    for bb_slug, root in M.GEN_ROOTS:
        if bb_filter and bb_slug != bb_filter:
            continue
        src = root / bb_slug
        if not src.exists():
            print(f"[skip] {src} (missing)")
            continue

        ta = time.time()
        files = sorted(src.glob("*.json"))
        ta = time.time() - ta

        tb, tc, td, te, tf, tg = [], [], [], [], [], []
        for p in files:
            t = time.time(); text = p.read_text(); tb.append(time.time() - t)
            try:
                t = time.time(); d = json.loads(text); tc.append(time.time() - t)
            except Exception:
                continue

            # D: full _answer_blob
            t = time.time(); blob = M._answer_blob(d); td.append(time.time() - t)
            # E: isolate the json.dumps(parsed) line
            parsed = d.get("parsed")
            if isinstance(parsed, dict):
                t = time.time(); _ = json.dumps(parsed, default=str); te.append(time.time() - t)
            # F: _selected_passages_text
            t = time.time(); _ = M._selected_passages_text(d); tf.append(time.time() - t)
            # G: dict insert
            case_id = d.get("case_id") or p.stem
            selected = d.get("selected_doc") or d.get("selected_doc_id") or ""
            t = time.time()
            if selected:
                key = (case_id, bb_slug)
                if key not in seen:
                    seen[key] = (case_id, bb_slug, selected, blob)
            tg.append(time.time() - t)

        n = len(files)
        print(f"\n=== {bb_slug}  ({src.name})  n={n} ===")
        print(f"  A glob+sort      total={ta:.3f}s")
        print(f"  B read_text      total={sum(tb):.3f}s  p50={median(tb)*1000:.2f}ms  p99={p99(tb)*1000:.2f}ms")
        print(f"  C json.loads     total={sum(tc):.3f}s  p50={median(tc)*1000:.2f}ms  p99={p99(tc)*1000:.2f}ms")
        print(f"  D _answer_blob   total={sum(td):.3f}s  p50={median(td)*1000:.2f}ms  p99={p99(td)*1000:.2f}ms")
        if te:
            print(f"  E dumps(parsed)  total={sum(te):.3f}s  p50={median(te)*1000:.2f}ms  p99={p99(te)*1000:.2f}ms  n_with_parsed={len(te)}")
        print(f"  F _selected_pass total={sum(tf):.3f}s  p50={median(tf)*1000:.2f}ms")
        print(f"  G dict insert    total={sum(tg):.3f}s")

    print(f"\nGRAND TOTAL: {time.time() - grand_total:.2f}s  seen={len(seen)}")


if __name__ == "__main__":
    bb = sys.argv[1] if len(sys.argv) > 1 else "Qwen__Qwen2.5-3B-Instruct"
    print(f"Bisecting backbone filter: {bb!r}")
    bisect(bb)
