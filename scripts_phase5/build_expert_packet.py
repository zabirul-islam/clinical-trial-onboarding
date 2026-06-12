#!/usr/bin/env python3
"""
Phase 5 Task 2 — blinded clinical-expert review packet builder.

Selects 30 V-final generations, stratified by EXISTING phase-4 properties only
(selector regime x semantic-flag status x backbone), builds blinded per-case
dossiers, a scoring sheet, and a gitignored blind key. With --add-baselines,
appends matching B1/B4 dossiers for the same cases (unscored by judges at
assembly time — by design; see plan Step 2.1).

Stratification design (frozen):
  - Backbones: Qwen-2.5-7B (best) + Qwen-2.5-3B (low-latency candidate), 15 each.
  - Within each backbone: all available either-judge semantic-flagged cases
    (consensus first), topped up with clean cases balanced across
    accept/abstain regime.
  - Seed 20260612 (reproducible).

Outputs (outputs/expert_review/):
  selection.csv                    case_id x backbone x stratum (NOT sent to experts)
  blind_key.csv                    blind_id -> (case_id, backbone, system)  [GITIGNORED]
  dossiers/<blind_id>.md           question + evidence + response, no model names
  scoring_sheet.csv                one row per blind_id, empty score columns

Run:
  python scripts_phase5/build_expert_packet.py
  python scripts_phase5/build_expert_packet.py --add-baselines   # after Task 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "expert_review"
SEED = 20260612
BACKBONES = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
PER_BACKBONE = 15

GEN_ROOTS = {
    "curated": ROOT / "outputs" / "backbone_gens",
    "expansion": ROOT / "outputs" / "phase4" / "n100_expansion" / "gens",
}
BASELINE_ROOT = ROOT / "outputs" / "phase5" / "baseline_gens"
BASELINE_IDS = ["B1_multi_rag", "B4_top1"]

SCORE_COLUMNS = [
    "factuality_1to5", "groundedness_1to5", "abstain_appropriateness_1to5",
    "safety_1to5", "patient_utility_1to5",
    "cross_trial_leak_yes_no",
    "failure_type_T1_T2_T3_or_none",
    "comments",
]


def blind_id(case_id: str, backbone: str, system: str) -> str:
    return hashlib.sha1(f"phase5-expert:{system}:{case_id}:{backbone}".encode()).hexdigest()[:10]


def load_gen(case_id: str, backbone: str) -> dict:
    slug = backbone.replace("/", "__")
    for root in GEN_ROOTS.values():
        fp = root / slug / f"{case_id}.json"
        if fp.exists():
            return json.loads(fp.read_text())
    raise FileNotFoundError(f"{case_id} / {backbone}")


def response_block(rec: dict) -> str:
    parsed = rec.get("parsed") or {}
    lines = [
        f"**Eligibility decision:** {rec.get('decision', '')}",
        f"**Patient-facing answer:** {rec.get('patient_facing_answer') or parsed.get('patient_facing_answer', '')}",
        f"**Supported patient facts:** {parsed.get('supported_patient_facts', [])}",
        f"**Missing patient facts:** {rec.get('missing_patient_facts', [])}",
        f"**Unresolved study requirements:** {rec.get('unresolved_study_requirements', [])}",
        f"**Safety note:** {parsed.get('safety_note', '')}",
    ]
    return "\n\n".join(lines)


def write_dossier(bid: str, rec: dict) -> None:
    evidence = str(rec.get("evidence", ""))[:6000]
    md = (
        f"# Case {bid}\n\n"
        f"## Patient question\n\n{rec.get('question','')}\n\n"
        f"## Evidence shown to the system\n\n```\n{evidence}\n```\n\n"
        f"## System response\n\n{response_block(rec)}\n"
    )
    (OUT / "dossiers" / f"{bid}.md").write_text(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-baselines", action="store_true")
    args = ap.parse_args()

    (OUT / "dossiers").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    raw = pd.read_csv(ROOT / "outputs/phase4/n114_aggregate/raw_n114.csv")
    sem = pd.read_csv(ROOT / "outputs/phase4/reviewer_fixes/semantic_leak_judge_long.csv")
    sem["backbone"] = sem["backbone"].str.replace("__", "/", n=1)
    flags = (sem.groupby(["case_id", "backbone"])["semantic_leak"]
             .agg(["sum", "count"]).reset_index())
    flags["either"] = flags["sum"] >= 1
    flags["consensus"] = flags["sum"] >= 2

    rows = []
    for bb in BACKBONES:
        sub = raw[raw["backbone"] == bb].merge(
            flags[["case_id", "backbone", "either", "consensus"]],
            on=["case_id", "backbone"], how="left").fillna({"either": False, "consensus": False})
        flagged = sub[sub["either"]].sort_values(["consensus", "case_id"],
                                                 ascending=[False, True])
        clean = sub[~sub["either"]]
        take = flagged.head(PER_BACKBONE // 2).copy()
        need = PER_BACKBONE - len(take)
        # top-up: balance accept/abstain among clean
        acc = clean[clean["regime_gate"] == "accept"]["case_id"].tolist()
        abst = clean[clean["regime_gate"] == "abstain"]["case_id"].tolist()
        rng.shuffle(acc), rng.shuffle(abst)
        topup_ids, i = [], 0
        while len(topup_ids) < need and (acc or abst):
            src = acc if (i % 2 == 0 and acc) or not abst else abst
            topup_ids.append(src.pop())
            i += 1
        topup = clean[clean["case_id"].isin(topup_ids)]
        sel = pd.concat([take, topup]).head(PER_BACKBONE)
        for _, r in sel.iterrows():
            rows.append({"case_id": r["case_id"], "backbone": bb, "system": "V-final",
                         "regime": r["regime_gate"],
                         "flag": "consensus" if r["consensus"] else
                                 ("either" if r["either"] else "clean")})

    sel_df = pd.DataFrame(rows)

    if args.add_baselines:
        base_rows = []
        for _, r in sel_df[sel_df["system"] == "V-final"].iterrows():
            for b in BASELINE_IDS:
                fp = BASELINE_ROOT / b / r["backbone"].replace("/", "__") / f"{r['case_id']}.json"
                if fp.exists():
                    base_rows.append({"case_id": r["case_id"], "backbone": r["backbone"],
                                      "system": b, "regime": r["regime"], "flag": "n/a"})
        sel_df = pd.concat([sel_df, pd.DataFrame(base_rows)], ignore_index=True)

    # blind ids + dossiers
    key_rows = []
    for _, r in sel_df.iterrows():
        bid = blind_id(r["case_id"], r["backbone"], r["system"])
        if r["system"] == "V-final":
            rec = load_gen(r["case_id"], r["backbone"])
        else:
            fp = (BASELINE_ROOT / r["system"] / r["backbone"].replace("/", "__")
                  / f"{r['case_id']}.json")
            rec = json.loads(fp.read_text())
        write_dossier(bid, rec)
        key_rows.append({"blind_id": bid, **r.to_dict()})

    key = pd.DataFrame(key_rows)
    order = key["blind_id"].tolist()
    rng.shuffle(order)
    key["presentation_order"] = key["blind_id"].map({b: i + 1 for i, b in enumerate(order)})
    key.to_csv(OUT / "blind_key.csv", index=False)
    sel_df.to_csv(OUT / "selection.csv", index=False)

    sheet = pd.DataFrame({"order": sorted(key["presentation_order"]),
                          "blind_id": [b for b, _ in sorted(zip(key["blind_id"],
                                                                key["presentation_order"]),
                                                            key=lambda t: t[1])]})
    for c in SCORE_COLUMNS:
        sheet[c] = ""
    sheet.to_csv(OUT / "scoring_sheet.csv", index=False)

    print(f"[done] {len(key)} dossiers; strata:\n{sel_df.groupby(['system','regime','flag']).size()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
