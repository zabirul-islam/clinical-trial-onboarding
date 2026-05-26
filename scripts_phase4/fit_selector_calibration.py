"""
T1.4b — Fit a calibrated probability over the 4 selector signals.

Hypothesis: the trial-first selector's gating signals (ρ, τ, μ, κ) jointly
encode trial relevance even though τ alone is not a calibrated probability.

Method:
  - Load outputs/tables/selector_signals_cache.csv (150 rows, 4 selector signals).
  - Filter to rows with gold_nct populated (~50 rows).
  - Define y_true variants:
      any    : selected_in_qrels_any        (rel ≥ 1 in TREC qrels)
      strict : selected_in_qrels_strict     (rel == 2 in TREC qrels)
  - Temporal split:
      TRAIN: source ∈ {trec2021, paraphrase_trec2021}
      TEST : source ∈ {trec2022, paraphrase_trec2022}
  - Models:
      A. τ-only baseline (no fit; p_pred = trial_score_share clipped to [0,1])
      B. Platt scaling on τ (1-feature logistic regression with intercept)
      C. Joint LR over (τ, ρ, μ, κ) with standardization
  - Eval each on TEST: ECE (10-bin), Brier, Brier Skill Score vs base rate.
  - 5-fold CV on combined as robustness check.

Outputs:
  outputs/phase4/calibration_lr/calibration_lr_summary.json
  outputs/phase4/calibration_lr/calibration_lr_per_bin.csv
  outputs/phase4/calibration_lr/calibration_lr_coefficients.csv
  outputs/phase4/calibration_lr/reliability_diagram_lr.{pdf,png}

Usage:
  python scripts_phase4/fit_selector_calibration.py \\
      --selector-cache outputs/tables/selector_signals_cache.csv \\
      --out-dir outputs/phase4/calibration_lr

CPU only; ~3 sec.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

FEATURES = ["trial_score_share", "dominance_ratio", "raw_max_cross", "best_rank"]
Y_VARIANTS = [
    ("any", "selected_in_qrels_any"),
    ("strict", "selected_in_qrels_strict"),
]
TRAIN_SOURCES = {"trec2021", "paraphrase_trec2021"}
TEST_SOURCES = {"trec2022", "paraphrase_trec2022"}


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def expected_calibration_error(
    p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p_pred, bins) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    rows = []
    n_total = len(p_pred)
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        n_b = int(mask.sum())
        if n_b == 0:
            rows.append(
                {
                    "bin_lo": float(bins[b]),
                    "bin_hi": float(bins[b + 1]),
                    "n": 0,
                    "frac": 0.0,
                    "mean_p_pred": np.nan,
                    "mean_y_true": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_p = float(p_pred[mask].mean())
        mean_y = float(y_true[mask].mean())
        gap = abs(mean_p - mean_y)
        ece += (n_b / n_total) * gap
        rows.append(
            {
                "bin_lo": float(bins[b]),
                "bin_hi": float(bins[b + 1]),
                "n": n_b,
                "frac": n_b / n_total,
                "mean_p_pred": mean_p,
                "mean_y_true": mean_y,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def brier(p_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))


def bss(p_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Brier Skill Score vs constant base-rate predictor."""
    br = float(np.mean(y_true)) if len(y_true) else float("nan")
    naive = br * (1.0 - br)
    if naive <= 0:
        return float("nan")
    return 1.0 - (brier(p_pred, y_true) / naive)


def fit_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_scaler: bool = True,
) -> dict:
    """Fit logistic regression on train, eval on test, return metrics + coefs."""
    if use_scaler:
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)
        scaler_means = scaler.mean_.tolist()
        scaler_scales = scaler.scale_.tolist()
    else:
        Xtr = X_train
        Xte = X_test
        scaler_means = None
        scaler_scales = None

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(Xtr, y_train)
    p_test = model.predict_proba(Xte)[:, 1]
    p_train = model.predict_proba(Xtr)[:, 1]

    ece_tr, _ = expected_calibration_error(p_train, y_train, n_bins=10)
    ece_te, bin_df = expected_calibration_error(p_test, y_test, n_bins=10)

    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "base_rate_train": float(y_train.mean()),
        "base_rate_test": float(y_test.mean()),
        "ece_train": ece_tr,
        "ece_test": ece_te,
        "brier_train": brier(p_train, y_train),
        "brier_test": brier(p_test, y_test),
        "bss_train": bss(p_train, y_train),
        "bss_test": bss(p_test, y_test),
        "coefficients": dict(zip(FEATURES, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
        "scaler_means": scaler_means,
        "scaler_scales": scaler_scales,
        "p_test": p_test.tolist(),
        "y_test": y_test.tolist(),
        "bin_df_test": bin_df.to_dict(orient="records"),
    }


def cv_fit(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict:
    """5-fold CV: fit LR on each fold's train, eval on fold's test."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    eces, briers, bsss, base_rates = [], [], [], []
    for tr_idx, te_idx in kf.split(X):
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr = scaler.transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        try:
            m.fit(Xtr, y[tr_idx])
        except ValueError:
            # single-class fold: skip
            continue
        p = m.predict_proba(Xte)[:, 1]
        e, _ = expected_calibration_error(p, y[te_idx], n_bins=10)
        eces.append(e)
        briers.append(brier(p, y[te_idx]))
        bsss.append(bss(p, y[te_idx]))
        base_rates.append(float(y[te_idx].mean()))
    return {
        "n_folds": len(eces),
        "ece_mean": float(np.mean(eces)) if eces else float("nan"),
        "ece_std": float(np.std(eces)) if eces else float("nan"),
        "brier_mean": float(np.mean(briers)) if briers else float("nan"),
        "brier_std": float(np.std(briers)) if briers else float("nan"),
        "bss_mean": float(np.mean(bsss)) if bsss else float("nan"),
        "bss_std": float(np.std(bsss)) if bsss else float("nan"),
        "base_rate_mean": float(np.mean(base_rates)) if base_rates else float("nan"),
    }


def tau_only_baseline(p_pred: np.ndarray, y: np.ndarray) -> dict:
    ece, _ = expected_calibration_error(p_pred, y, n_bins=10)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "ece": ece,
        "brier": brier(p_pred, y),
        "bss": bss(p_pred, y),
    }


def plot_reliability(
    bin_dfs: dict[str, pd.DataFrame],
    out_path: Path,
    title: str = "Reliability — joint LR vs τ-only",
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
    colors = {"tau_only_test": "#d62728", "joint_lr_test": "#1f77b4"}
    for label, df in bin_dfs.items():
        valid = df.dropna(subset=["mean_p_pred", "mean_y_true"])
        ax.plot(
            valid["mean_p_pred"],
            valid["mean_y_true"],
            "o-",
            color=colors.get(label, "black"),
            markersize=8,
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("Predicted prob")
    ax.set_ylabel("Empirical fraction relevant")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selector-cache",
        type=str,
        default="outputs/tables/selector_signals_cache.csv",
    )
    ap.add_argument("--out-dir", type=str, default="outputs/phase4/calibration_lr")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.selector_cache)
    print(f"[load] {len(df)} rows from {args.selector_cache}")

    # Filter: gold_nct populated
    has_gold = df["gold_nct"].notna() & (df["gold_nct"].astype(str).str.startswith("NCT"))
    df = df[has_gold].reset_index(drop=True)
    print(f"[filter] kept {len(df)} rows with gold_nct populated")

    # Split by source
    train_mask = df["source"].isin(TRAIN_SOURCES)
    test_mask = df["source"].isin(TEST_SOURCES)
    print(f"[split] train (TREC 2021 family): n={int(train_mask.sum())}")
    print(f"[split] test  (TREC 2022 family): n={int(test_mask.sum())}")
    print(f"[source breakdown of gold-having cases] {df['source'].value_counts().to_dict()}")

    cv_only = int(test_mask.sum()) < 5
    if cv_only:
        print(
            "[mode] temporal test split has <5 rows → fall back to 5-fold CV ONLY mode"
        )
    elif int(train_mask.sum()) < 5:
        print("[warn] train split has <5 rows; CV mode recommended")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "n_total_with_gold": int(len(df)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "features": FEATURES,
        "y_variants": {},
    }

    coef_rows: list[dict] = []
    bin_csv_rows: list[dict] = []

    for label, col in Y_VARIANTS:
        print(f"\n=== y_true = {label} ({col}) ===")
        y_all = _to_bool(df[col]).astype(int).to_numpy()
        tau_all = df["trial_score_share"].astype(float).clip(0, 1).to_numpy()

        # τ-only baseline on whatever test we have
        if cv_only:
            # Use entire pool as the "test" for τ-only baseline
            tau_test = tau_all
            y_test = y_all
            tau_metrics = tau_only_baseline(tau_test, y_test)
            print(
                f"[τ-only ALL] n={tau_metrics['n']} base={tau_metrics['base_rate']:.3f} "
                f"ECE={tau_metrics['ece']:.4f} Brier={tau_metrics['brier']:.4f} "
                f"BSS={tau_metrics['bss']:+.3f}"
            )
        else:
            tau_test = tau_all[test_mask.to_numpy()]
            y_test = y_all[test_mask.to_numpy()]
            tau_metrics = tau_only_baseline(tau_test, y_test)
            print(
                f"[τ-only test] n={tau_metrics['n']} base={tau_metrics['base_rate']:.3f} "
                f"ECE={tau_metrics['ece']:.4f} Brier={tau_metrics['brier']:.4f} "
                f"BSS={tau_metrics['bss']:+.3f}"
            )

        # Joint LR
        joint = None
        cv = None

        if cv_only:
            # Pure 5-fold CV mode on the full gold-having pool
            X_all = df[FEATURES].to_numpy(dtype=float)
            y_combined = y_all
            if len(np.unique(y_combined)) >= 2:
                cv = cv_fit(X_all, y_combined, n_splits=5, seed=args.seed)
                print(
                    f"[5-fold CV] ECE={cv['ece_mean']:.4f}±{cv['ece_std']:.4f} "
                    f"Brier={cv['brier_mean']:.4f}±{cv['brier_std']:.4f} "
                    f"BSS={cv['bss_mean']:+.3f}±{cv['bss_std']:.3f} "
                    f"base={cv['base_rate_mean']:.3f}"
                )
                # Also fit on full pool to get coefficients (not for eval, just to report)
                from sklearn.linear_model import LogisticRegression as _LR
                from sklearn.preprocessing import StandardScaler as _SS
                _scaler = _SS().fit(X_all)
                _Xs = _scaler.transform(X_all)
                _model = _LR(C=1.0, max_iter=1000, solver="lbfgs").fit(_Xs, y_combined)
                joint_coefs = dict(zip(FEATURES, _model.coef_[0].tolist()))
                joint_intercept = float(_model.intercept_[0])
                print(f"  coefs (full-fit, for inspection only): {joint_coefs}")
                print(f"  intercept: {joint_intercept:+.3f}")
                for feat, c in joint_coefs.items():
                    coef_rows.append({"y_variant": label, "feature": feat, "coef": c})
                coef_rows.append(
                    {"y_variant": label, "feature": "(intercept)", "coef": joint_intercept}
                )
            else:
                print(f"[skip] 5-fold CV for {label}: single-class y on entire pool")
        else:
            # Original temporal split path
            X_train = df.loc[train_mask, FEATURES].to_numpy(dtype=float)
            y_train = y_all[train_mask.to_numpy()]
            X_test = df.loc[test_mask, FEATURES].to_numpy(dtype=float)
            y_test_lr = y_all[test_mask.to_numpy()]
            if len(np.unique(y_train)) < 2 or len(X_test) == 0:
                print(
                    f"[skip] joint LR temporal for {label}: insufficient train classes "
                    f"or empty test (n_train={len(y_train)}, n_test={len(X_test)})"
                )
            else:
                joint = fit_and_eval(X_train, y_train, X_test, y_test_lr, use_scaler=True)
                print(
                    f"[joint LR test] n={joint['n_test']} base={joint['base_rate_test']:.3f} "
                    f"ECE_train={joint['ece_train']:.4f} ECE_test={joint['ece_test']:.4f} "
                    f"Brier_test={joint['brier_test']:.4f} BSS_test={joint['bss_test']:+.3f}"
                )
                print(f"  coefs: {joint['coefficients']}")
                print(f"  intercept: {joint['intercept']:+.3f}")
                for feat, c in joint["coefficients"].items():
                    coef_rows.append({"y_variant": label, "feature": feat, "coef": c})
                coef_rows.append(
                    {"y_variant": label, "feature": "(intercept)", "coef": joint["intercept"]}
                )

            # CV on combined as robustness
            X_all = df.loc[train_mask | test_mask, FEATURES].to_numpy(dtype=float)
            y_combined = y_all[(train_mask | test_mask).to_numpy()]
            if len(np.unique(y_combined)) >= 2:
                cv = cv_fit(X_all, y_combined, n_splits=5, seed=args.seed)
                print(
                    f"[5-fold CV] ECE={cv['ece_mean']:.4f}±{cv['ece_std']:.4f} "
                    f"Brier={cv['brier_mean']:.4f}±{cv['brier_std']:.4f} "
                    f"BSS={cv['bss_mean']:+.3f}±{cv['bss_std']:.3f} "
                    f"base={cv['base_rate_mean']:.3f}"
                )

        summary["y_variants"][label] = {
            "y_true_col": col,
            "tau_only_test": tau_metrics,
            "joint_lr_test": (
                {k: v for k, v in joint.items() if k not in {"p_test", "y_test", "bin_df_test"}}
                if joint
                else None
            ),
            "cv_5fold_combined": cv,
        }

        # Persist per-bin reliability for the "any" headline plot
        if label == "any" and joint:
            tau_bins = expected_calibration_error(tau_test, y_test, n_bins=10)[1]
            joint_bins = pd.DataFrame(joint["bin_df_test"])
            plot_reliability(
                {"tau_only_test": tau_bins, "joint_lr_test": joint_bins},
                out_dir / "reliability_diagram_lr",
                title=f"Reliability on TEST (TREC 2022) — y={label} (rel≥1)",
            )

        # Per-bin csv
        if joint:
            for r in joint["bin_df_test"]:
                bin_csv_rows.append({"y_variant": label, "model": "joint_lr_test", **r})
        for _, r in expected_calibration_error(tau_test, y_test, n_bins=10)[1].iterrows():
            bin_csv_rows.append({"y_variant": label, "model": "tau_only_test", **r.to_dict()})

    # Persist
    pd.DataFrame(bin_csv_rows).to_csv(out_dir / "calibration_lr_per_bin.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out_dir / "calibration_lr_coefficients.csv", index=False)
    with (out_dir / "calibration_lr_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[wrote] {out_dir}/calibration_lr_*.{{json,csv}} + reliability_diagram_lr.*")

    # Headline verdict
    print("\n=== HEADLINE VERDICT ===")
    for label, _ in Y_VARIANTS:
        s = summary["y_variants"].get(label, {})
        tau = s.get("tau_only_test", {}) or {}
        joint = s.get("joint_lr_test") or {}
        cv_5 = s.get("cv_5fold_combined") or {}

        tau_bss = tau.get("bss", float("nan"))

        if cv_only:
            # Use CV BSS as the primary number
            joint_bss = cv_5.get("bss_mean", float("nan"))
            joint_ece = cv_5.get("ece_mean", float("nan"))
            tau_ece = tau.get("ece", float("nan"))
            verdict = "WIN" if (joint_bss == joint_bss and joint_bss > 0) else "STILL_NEGATIVE"
            print(
                f"{label:>6s} (CV5): τ-only BSS={tau_bss:+.3f} ECE={tau_ece:.4f}  |  "
                f"joint LR CV BSS={joint_bss:+.3f} ECE={joint_ece:.4f}  →  {verdict}"
            )
        else:
            if not joint:
                continue
            joint_bss = joint.get("bss_test", float("nan"))
            joint_ece = joint.get("ece_test", float("nan"))
            tau_ece = tau.get("ece", float("nan"))
            delta_bss = joint_bss - tau_bss
            delta_ece = tau_ece - joint_ece
            verdict = "WIN" if joint_bss > 0 else "STILL_NEGATIVE"
            print(
                f"{label:>6s}: τ-only BSS={tau_bss:+.3f}, "
                f"joint LR BSS={joint_bss:+.3f}  "
                f"(ΔBSS={delta_bss:+.3f}, ΔECE={delta_ece:+.4f})  "
                f"→ {verdict}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
