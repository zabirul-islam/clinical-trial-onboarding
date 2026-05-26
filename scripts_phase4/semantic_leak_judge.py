"""
Phase 4 — 3rd leak detector: LLM-judge semantic cross-trial leakage.

Motivation
----------
The two original detectors (narrow NCT-regex + wide entity-resolved token-match)
are *lexical*. They miss paraphrased semantic leakage — e.g. the answer about
trial t* describes procedures or dosing that only appear in a non-selected
candidate trial t' in vocabulary that lacks any distinctive token.

This script applies an LLM-judge semantic detector to every (case x backbone)
generation. The judge sees:
  * the patient-facing answer fields,
  * the selected trial t*'s passages (already shown to the generator),
  * the *non-selected* candidate trial passages (top 1-2 per non-selected
    doc from the same retrieval pool).

It returns a binary flag and a brief rationale. Two judges (Sonnet-4.5,
GPT-4o) provide IRR; we report either-judge and both-judges rates separately.

Schema written (one JSONL line per case x backbone x judge)
-----------------------------------------------------------
  {
    "judge": "sonnet" | "gpt4o",
    "judge_model": "<model id>",
    "blind_id": "<sha1[:12]>",
    "case_id": "...",
    "backbone": "...",
    "selected_doc": "NCT...",
    "n_candidates_seen": <int>,
    "semantic_leak": 0 | 1,
    "suspect_doc": "NCT..." | null,
    "claim_excerpt": "<<= 250 char>" | null,
    "rationale": "<2 sentences>",
    "latency_sec": <float>
  }

Resumable: skips (case x backbone x judge) already written successfully.
Budget: ~684 x 2 ≈ 1368 calls at ~3k-in / 200-out ≈ $6-10.

Usage
-----
  export ANTHROPIC_API_KEY=...
  export OPENAI_API_KEY=...
  python scripts_phase4/semantic_leak_judge.py \
      --backbones Qwen__Qwen2.5-3B-Instruct \
      --judges sonnet gpt4o \
      --out-dir outputs/phase4/reviewer_fixes \
      --max-cases 30                                  # smoke test
  python scripts_phase4/semantic_leak_judge.py        # full run, all backbones, both judges
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None
try:
    import openai  # type: ignore
except Exception:
    openai = None

REPO = Path("/Users/zabir/Desktop/Clinical-Trial")


def _prewarm_icloud(paths, *, label: str = "prewarm") -> None:
    """Force iCloud Drive to materialize cloud-stub files/dirs.
    macOS-only. No-op if brctl missing, paths absent, or non-iCloud.
    Errors swallowed (best-effort)."""
    import shutil
    import subprocess
    brctl = shutil.which("brctl")
    if not brctl:
        return
    real = [Path(p) for p in paths if p and Path(p).exists()]
    if not real:
        return
    t0 = time.time()
    for p in real:
        try:
            subprocess.run(
                [brctl, "download", str(p)],
                check=False, capture_output=True, timeout=600,
            )
        except Exception:
            pass
    dt = time.time() - t0
    if dt > 0.5:
        print(f"[{label}] iCloud pre-warm {len(real)} paths in {dt:.1f}s")

# ──────────────────────────────────────────────────────────────
# Prompt — semantic cross-trial leakage detector
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You audit clinical-trial onboarding responses for CROSS-TRIAL semantic "
    "leakage. A leak occurs when the answer states a factual claim "
    "(eligibility criterion, procedure, dose, schedule, intervention, outcome, "
    "or sponsor detail) that is supported ONLY by a non-selected trial in "
    "the retrieval pool, not by the selected trial t*. Paraphrase counts as "
    "leakage. Generic clinical English that fits any trial is NOT leakage. "
    "Return JSON only."
)

USER_TEMPLATE = """\
## Task
Decide whether the Assistant's answer contains any factual claim about a
clinical trial that is supported ONLY by a non-selected candidate trial
(i.e., a trial whose NCT id is NOT {selected_doc}), not by the selected
trial t* = {selected_doc}. Paraphrase counts; lexical match is not required.

If the claim is supported by t* (even partially) OR is generic clinical
English ("you may have to come in for visits", "a study doctor will explain
the consent"), it is NOT a leak.

## Selected trial t* = {selected_doc}
{selected_passages}

## Non-selected candidate trials in the retrieval pool ({n_candidates} shown)
{candidate_passages}

## Assistant's answer (fields the patient would see)
{answer_blob}

## Output — JSON only, no markdown fences, no prose
{{
  "semantic_leak": 0 or 1,
  "suspect_doc": "<NCT id of the non-selected trial whose content was leaked, or null>",
  "claim_excerpt": "<<=250 chars verbatim from the answer that constitutes the leak, or null>",
  "rationale": "<2 sentences citing the specific claim and which non-selected trial supports it>"
}}
"""

REQUIRED_KEYS = ["semantic_leak", "suspect_doc", "claim_excerpt", "rationale"]


def parse_judge_json(txt: str, judge: str) -> Dict[str, Any]:
    raw = txt.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"[{judge}] no JSON found in: {txt[:300]}")
    obj = json.loads(m.group(0))
    for k in REQUIRED_KEYS:
        if k not in obj:
            obj[k] = None if k != "semantic_leak" else 0
    try:
        obj["semantic_leak"] = int(obj["semantic_leak"])
    except Exception:
        obj["semantic_leak"] = 0
    if obj["semantic_leak"] not in (0, 1):
        obj["semantic_leak"] = 0
    return obj


# ──────────────────────────────────────────────────────────────
# Judges (clone of run_llm_judge.py pattern)
# ──────────────────────────────────────────────────────────────
def _retry_delay(attempt: int, err: Exception) -> float:
    return min(60.0, 2.0 * (2 ** attempt)) + random.random() * 2.0


async def _with_retry(coro_factory, *, judge_name: str, max_attempts: int = 8):
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retry = any(s in msg for s in (
                "429", "rate_limit", "rate limit", "overload", "timeout",
                "connection",
            ))
            if not retry or attempt == max_attempts - 1:
                raise
            delay = _retry_delay(attempt, e)
            print(f"  [retry] {judge_name} attempt {attempt+1}/{max_attempts} "
                  f"after {delay:.1f}s — {type(e).__name__}")
            await asyncio.sleep(delay)
    assert last_err is not None
    raise last_err


class RateGate:
    def __init__(self, interval: float):
        self.interval = float(interval)
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
            self._next = max(time.monotonic(), self._next) + self.interval


class Judge:
    name: str
    model: str
    gate: Optional[RateGate] = None

    async def score(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class SonnetJudge(Judge):
    name = "sonnet"
    model = "claude-sonnet-4-5-20250929"

    def __init__(self, min_interval: float = 8.0):
        if anthropic is None:
            raise RuntimeError("pip install anthropic")
        self.client = anthropic.AsyncAnthropic()
        self.gate = RateGate(min_interval)

    async def score(self, prompt: str) -> Dict[str, Any]:
        async def _call():
            if self.gate is not None:
                await self.gate.wait()
            return await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        resp = await _with_retry(_call, judge_name=self.name)
        txt = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        )
        return parse_judge_json(txt, judge=self.name)


class GPT4oJudge(Judge):
    name = "gpt4o"
    model = "gpt-4o-2024-11-20"

    def __init__(self, min_interval: float = 8.0):
        if openai is None:
            raise RuntimeError("pip install openai>=1.0")
        self.client = openai.AsyncOpenAI()
        self.gate = RateGate(min_interval)

    async def score(self, prompt: str) -> Dict[str, Any]:
        async def _call():
            if self.gate is not None:
                await self.gate.wait()
            return await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
        resp = await _with_retry(_call, judge_name=self.name)
        txt = resp.choices[0].message.content or ""
        return parse_judge_json(txt, judge=self.name)


JUDGE_REGISTRY = {"sonnet": SonnetJudge, "gpt4o": GPT4oJudge}


# ──────────────────────────────────────────────────────────────
# Candidate-pool reconstruction
# ──────────────────────────────────────────────────────────────
_CORPUS: Optional[pd.DataFrame] = None
_CORPUS_INDEX: Optional[pd.DataFrame] = None  # doc_id-indexed for fast lookup
_BM25: Optional[pd.DataFrame] = None
_SSC: Optional[pd.DataFrame] = None


def _corpus() -> pd.DataFrame:
    """Return the corpus DataFrame indexed by doc_id for O(log n) lookup."""
    global _CORPUS, _CORPUS_INDEX
    if _CORPUS_INDEX is None:
        path = REPO / "processed" / "trial_evidence_passages.parquet"
        _prewarm_icloud([path], label="corpus")
        df = pd.read_parquet(
            path,
            columns=["doc_id", "section", "passage_text", "passage_id"],
        )
        # Sort + set index for O(log n) lookups by doc_id
        df = df.sort_values("doc_id").set_index("doc_id", drop=False)
        _CORPUS_INDEX = df
        _CORPUS = df  # backwards compatibility
    return _CORPUS_INDEX


def _bm25() -> pd.DataFrame:
    global _BM25
    if _BM25 is None:
        path = REPO / "outputs" / "tables" / "bm25_full_text_top100_candidates.csv"
        _prewarm_icloud([path], label="bm25")
        _BM25 = pd.read_csv(path)
    return _BM25


def _ssc() -> pd.DataFrame:
    global _SSC
    if _SSC is None:
        path = REPO / "outputs" / "tables" / "selector_signals_cache.csv"
        _prewarm_icloud([path], label="ssc")
        _SSC = pd.read_csv(path)
    return _SSC


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _passages_for_doc(doc_id: str, k: int = 2,
                      max_chars: int = 700) -> List[str]:
    """Return up to k short passages for a trial doc_id, preferring
    eligibility + brief_summary + detailed_description in that order.
    Uses an indexed corpus for O(log n) lookup."""
    corp = _corpus()
    try:
        df = corp.loc[[doc_id]]
    except KeyError:
        return []
    if df.empty:
        return []
    order = {"eligibility": 0, "brief_summary": 1, "summary": 1,
             "detailed_description": 2, "primary_outcome": 3,
             "secondary_outcome": 4}
    df = df.assign(_pri=df["section"].map(order).fillna(9))
    df = df.sort_values(["_pri", "passage_id"]).head(k)
    return [_truncate(t, max_chars) for t in df["passage_text"].tolist()]


def candidate_pool_for_case(case_id: str, selected_doc: str,
                            k_docs: int = 4) -> List[Dict[str, Any]]:
    """Return list of {'doc_id', 'passages'} for top-k non-selected docs in
    the retrieval pool for this case. Two routes:
      (1) handcrafted case_01..case_15 → per-case grounding_evidence csv
      (2) TREC/paraphrase cases → BM25 top-100 via trec_query_id
    """
    selected = (selected_doc or "").upper()

    # Route 1: per-case retrieval dir (handcrafted seed cases)
    for run_dir in ("eval_runs_final", "eval_runs_v2"):
        p = (REPO / "outputs" / run_dir / case_id /
             "grounding_evidence_top_passages_raw.csv")
        if p.exists():
            df = pd.read_csv(p)
            df = df[df["doc_id"].str.upper() != selected]
            if df.empty:
                return []
            # collapse to one row per doc, keep best cross_score
            df = (df.sort_values("cross_score", ascending=False)
                    .drop_duplicates("doc_id")
                    .head(k_docs))
            out = []
            for _, r in df.iterrows():
                out.append({
                    "doc_id": r["doc_id"],
                    "passages": [_truncate(str(r["passage_text"]), 700)]
                                + _passages_for_doc(r["doc_id"], k=1),
                })
            # dedupe passages per doc
            for c in out:
                seen, dedup = set(), []
                for t in c["passages"]:
                    h = t[:80]
                    if h in seen:
                        continue
                    seen.add(h); dedup.append(t)
                c["passages"] = dedup[:2]
            return out

    # Route 2: BM25 top-100 via trec_query_id from selector_signals_cache
    ssc = _ssc()
    row = ssc[ssc["case_id"] == case_id]
    if row.empty:
        return []
    qid = row.iloc[0].get("trec_query_id")
    if pd.isna(qid):
        return []
    qid = int(qid)
    bm = _bm25()
    pool = bm[(bm["query_id"] == qid) &
              (bm["doc_id"].str.upper() != selected)]
    pool = pool.sort_values("rank_bm25").head(k_docs)
    out = []
    for _, r in pool.iterrows():
        passages = _passages_for_doc(r["doc_id"], k=2)
        if not passages:
            continue
        out.append({"doc_id": r["doc_id"], "passages": passages})
    return out


# ──────────────────────────────────────────────────────────────
# Per-gen ingestion
# ──────────────────────────────────────────────────────────────
GEN_ROOTS = [
    ("Qwen__Qwen2.5-3B-Instruct", REPO / "outputs" / "backbone_gens"),
    ("Qwen__Qwen2.5-7B-Instruct", REPO / "outputs" / "backbone_gens"),
    ("meta-llama__Meta-Llama-3.1-8B-Instruct",
     REPO / "outputs" / "backbone_gens"),
    ("mistralai__Mistral-7B-Instruct-v0.3",
     REPO / "outputs" / "backbone_gens"),
    ("Qwen__Qwen2.5-3B-Instruct",
     REPO / "outputs" / "phase4" / "n100_expansion" / "gens"),
    ("Qwen__Qwen2.5-7B-Instruct",
     REPO / "outputs" / "phase4" / "n100_expansion" / "gens"),
    ("meta-llama__Meta-Llama-3.1-8B-Instruct",
     REPO / "outputs" / "phase4" / "n100_expansion" / "gens"),
    ("mistralai__Mistral-7B-Instruct-v0.3",
     REPO / "outputs" / "phase4" / "n100_expansion" / "gens"),
    ("gpt-4o",
     REPO / "outputs" / "phase4" / "zeroshot_baseline" / "gens"),
    ("claude-sonnet-4-5-20250929",
     REPO / "outputs" / "phase4" / "zeroshot_baseline" / "gens"),
]


@dataclass
class GenRecord:
    case_id: str
    backbone: str
    selected_doc: str
    answer_blob: str
    selected_passages: str
    raw_path: str


def _answer_blob(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ("patient_facing_answer", "raw_generation"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    parsed = d.get("parsed")
    if isinstance(parsed, dict):
        parts.append(json.dumps(parsed, default=str))
    for k in ("eligibility", "consent", "explanation"):
        v = d.get(k)
        if v is not None:
            parts.append(json.dumps(v, default=str))
    blob = "\n\n".join(parts)
    return _truncate(blob, 5000)


def _selected_passages_text(d: Dict[str, Any]) -> str:
    p = d.get("passages")
    if isinstance(p, list) and p:
        rows = []
        for x in p[:6]:
            if isinstance(x, dict):
                rows.append(
                    f"[{x.get('section', '')}] "
                    f"{_truncate(str(x.get('passage_text', '')), 700)}"
                )
        if rows:
            return "\n".join(rows)
    ev = d.get("evidence")
    if isinstance(ev, str):
        return _truncate(ev, 4000)
    return ""


def discover_records(backbones: Optional[List[str]] = None,
                     max_cases: Optional[int] = None) -> List[GenRecord]:
    # Pre-warm iCloud cloud-stub dirs before per-file read_text():
    # gen roots (8 of 10 are fileprovider-backed) + Route-1 candidate dirs.
    warm_paths = [
        root / bb for bb, root in GEN_ROOTS
        if (root / bb).exists()
        and (not backbones or bb in backbones)
    ] + [
        REPO / "outputs" / "eval_runs_final",
        REPO / "outputs" / "eval_runs_v2",
    ]
    _prewarm_icloud(warm_paths, label="discover")

    seen: Dict[tuple, GenRecord] = {}
    for bb_slug, root in GEN_ROOTS:
        if backbones and bb_slug not in backbones:
            continue
        src = root / bb_slug
        if not src.exists():
            continue
        files = sorted(src.glob("*.json"))
        for p in files:
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            case_id = d.get("case_id") or p.stem
            selected = (d.get("selected_doc") or d.get("selected_doc_id")
                        or "")
            if not selected:
                continue
            key = (case_id, bb_slug)
            if key in seen:
                continue
            seen[key] = GenRecord(
                case_id=case_id,
                backbone=bb_slug,
                selected_doc=selected,
                answer_blob=_answer_blob(d),
                selected_passages=_selected_passages_text(d),
                raw_path=str(p),
            )
    recs = list(seen.values())
    if max_cases:
        recs = recs[:max_cases]
    print(f"[discover] {len(recs)} (case x backbone) records")
    return recs


# ──────────────────────────────────────────────────────────────
# Prompt building + I/O
# ──────────────────────────────────────────────────────────────
def build_prompt(rec: GenRecord,
                 candidates: List[Dict[str, Any]]) -> str:
    cand_blocks = []
    for c in candidates:
        block = f"### Candidate trial {c['doc_id']}\n"
        for txt in c["passages"]:
            block += f"- {txt}\n"
        cand_blocks.append(block)
    if not cand_blocks:
        cand_blocks = ["(no non-selected candidates available — "
                       "single-trial pool by retrieval)"]
    return USER_TEMPLATE.format(
        selected_doc=rec.selected_doc,
        selected_passages=rec.selected_passages or "(empty)",
        n_candidates=len(candidates),
        candidate_passages="\n".join(cand_blocks),
        answer_blob=rec.answer_blob or "(empty)",
    )


def stable_blind_id(rec: GenRecord, salt: str = "phase4-semleak") -> str:
    h = hashlib.sha1(
        f"{salt}:{rec.case_id}:{rec.backbone}".encode()
    ).hexdigest()
    return h[:12]


def load_existing(jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl.exists():
        return done
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "error" in d:
            continue
        if "semantic_leak" not in d:
            continue
        done.add(f"{d['case_id']}|{d['backbone']}")
    return done


def compact_jsonl(jsonl: Path) -> int:
    if not jsonl.exists():
        return 0
    keep: Dict[str, str] = {}
    dropped = 0
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            dropped += 1
            continue
        if "error" in d or "semantic_leak" not in d:
            dropped += 1
            continue
        keep[f"{d['case_id']}|{d['backbone']}"] = line
    jsonl.write_text("\n".join(keep.values()) + ("\n" if keep else ""))
    return dropped


# ──────────────────────────────────────────────────────────────
# Async runner
# ──────────────────────────────────────────────────────────────
async def run_judge(judge: Judge, records: List[GenRecord], out_path: Path,
                    concurrency: int, seed: int,
                    candidate_cache: Dict[str, List[Dict[str, Any]]]) -> int:
    done = load_existing(out_path)
    todo = [r for r in records if f"{r.case_id}|{r.backbone}" not in done]
    print(f"[{judge.name}] {len(todo)}/{len(records)} to score "
          f"(resume: {len(done)} done)")
    if not todo:
        return 0
    random.Random(seed).shuffle(todo)
    sem = asyncio.Semaphore(concurrency)
    fout = out_path.open("a", encoding="utf-8")
    lock = asyncio.Lock()
    counter = {"ok": 0, "err": 0, "leak": 0}
    t0 = time.time()

    async def worker(rec: GenRecord):
        cands = candidate_cache.get(rec.case_id) or []
        prompt = build_prompt(rec, cands)
        async with sem:
            try:
                t_start = time.time()
                out = await judge.score(prompt)
                dt = time.time() - t_start
                line = {
                    "judge": judge.name,
                    "judge_model": judge.model,
                    "blind_id": stable_blind_id(rec),
                    "case_id": rec.case_id,
                    "backbone": rec.backbone,
                    "selected_doc": rec.selected_doc,
                    "n_candidates_seen": len(cands),
                    "semantic_leak": int(out.get("semantic_leak") or 0),
                    "suspect_doc": out.get("suspect_doc"),
                    "claim_excerpt": _truncate(
                        str(out.get("claim_excerpt") or ""), 250
                    ) if out.get("claim_excerpt") else None,
                    "rationale": _truncate(
                        str(out.get("rationale") or ""), 600
                    ),
                    "latency_sec": round(dt, 2),
                }
                async with lock:
                    fout.write(json.dumps(line) + "\n")
                    fout.flush()
                    counter["ok"] += 1
                    counter["leak"] += line["semantic_leak"]
                    if counter["ok"] % 20 == 0:
                        print(f"  [{judge.name}] {counter['ok']} done, "
                              f"{counter['leak']} flagged")
            except Exception as e:
                counter["err"] += 1
                async with lock:
                    fout.write(json.dumps({
                        "judge": judge.name,
                        "case_id": rec.case_id,
                        "backbone": rec.backbone,
                        "error": f"{type(e).__name__}: {e}",
                    }) + "\n")
                    fout.flush()
                print(f"  [!] {judge.name} {rec.case_id} {rec.backbone}: {e}")

    await asyncio.gather(*(worker(r) for r in todo))
    fout.close()
    dt = time.time() - t0
    print(f"[{judge.name}] done — ok={counter['ok']} err={counter['err']} "
          f"leak={counter['leak']} in {dt:.1f}s")
    return counter["err"]


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
def build_summary(out_dir: Path) -> None:
    import numpy as np
    rows = []
    for jsonl in sorted(out_dir.glob("semantic_leak_judge_*.jsonl")):
        judge = jsonl.stem.replace("semantic_leak_judge_", "")
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "error" in d or "semantic_leak" not in d:
                continue
            rows.append({
                "judge": judge,
                "case_id": d["case_id"],
                "backbone": d["backbone"],
                "semantic_leak": d["semantic_leak"],
            })
    if not rows:
        print("[summary] no rows")
        return
    df = pd.DataFrame(rows)
    long_path = out_dir / "semantic_leak_judge_long.csv"
    df.to_csv(long_path, index=False)
    print(f"[summary] wrote long → {long_path}")

    # per-backbone rates — handle single- or dual-judge runs
    piv = (df.pivot_table(index=["case_id", "backbone"],
                          columns="judge",
                          values="semantic_leak",
                          aggfunc="max")
             .reset_index())
    judges_present = [j for j in ("sonnet", "gpt4o") if j in piv.columns]
    if len(judges_present) == 2:
        piv["either"] = piv[judges_present].max(axis=1)
        piv["both"]   = piv[judges_present].min(axis=1)
    agg_spec = {"n": ("case_id", "count")}
    for j in judges_present:
        agg_spec[f"{j}_rate"] = (j, "mean")
    if len(judges_present) == 2:
        agg_spec["either_rate"] = ("either", "mean")
        agg_spec["both_rate"]   = ("both", "mean")
    summ = piv.groupby("backbone").agg(**agg_spec).reset_index()
    summ_path = out_dir / "semantic_leak_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"[summary] wrote per-backbone summary → {summ_path}")
    print(summ.to_string(index=False))

    # Cohen kappa across all (case, backbone) where both judges scored
    if len(judges_present) == 2 and {"sonnet", "gpt4o"}.issubset(piv.columns):
        m = piv.dropna(subset=["sonnet", "gpt4o"])
        if len(m) >= 5:
            a = m["sonnet"].astype(int).values
            b = m["gpt4o"].astype(int).values
            po = float((a == b).mean())
            # marginal probabilities
            pa1 = float(a.mean()); pb1 = float(b.mean())
            pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
            kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
            kappa_path = out_dir / "semantic_leak_irr.csv"
            pd.DataFrame([{
                "n_pairs": len(m),
                "agree_rate": po,
                "sonnet_rate": pa1,
                "gpt4o_rate": pb1,
                "cohen_kappa": kappa,
            }]).to_csv(kappa_path, index=False)
            print(f"[summary] wrote IRR → {kappa_path}  "
                  f"(κ={kappa:.3f}, agree={po:.3f})")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=None,
                    help="Filter to specific backbone slugs.")
    ap.add_argument("--judges", nargs="+",
                    default=["sonnet", "gpt4o"],
                    choices=list(JUDGE_REGISTRY.keys()))
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs" / "phase4" / "reviewer_fixes")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Cap number of (case x backbone) records "
                         "(smoke tests).")
    ap.add_argument("--k-candidates", type=int, default=4,
                    help="Top-k non-selected docs per case.")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min-interval", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip API; rebuild CSV summaries from existing JSONL.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        build_summary(args.out_dir)
        return 0

    records = discover_records(backbones=args.backbones,
                               max_cases=args.max_cases)
    if not records:
        print("[!] no records found")
        return 2

    # Pre-build candidate-pool cache once (saves repeated parquet scans)
    print("[cache] building per-case candidate pools …")
    cand_cache: Dict[str, List[Dict[str, Any]]] = {}
    for case_id in sorted({r.case_id for r in records}):
        selected = next(r.selected_doc for r in records if r.case_id == case_id)
        cand_cache[case_id] = candidate_pool_for_case(
            case_id, selected, k_docs=args.k_candidates
        )
    no_cands = sum(1 for v in cand_cache.values() if not v)
    print(f"[cache] {len(cand_cache)} cases, "
          f"{no_cands} with empty non-selected pool")

    # Compact jsonl, run each judge
    errs = 0
    for jname in args.judges:
        out = args.out_dir / f"semantic_leak_judge_{jname}.jsonl"
        dropped = compact_jsonl(out)
        if dropped:
            print(f"[compact] {jname}: dropped {dropped} bad lines")

    for jname in args.judges:
        judge = JUDGE_REGISTRY[jname](min_interval=args.min_interval)
        out = args.out_dir / f"semantic_leak_judge_{jname}.jsonl"
        errs += asyncio.run(run_judge(
            judge, records, out, args.concurrency, args.seed, cand_cache
        ))

    build_summary(args.out_dir)
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
