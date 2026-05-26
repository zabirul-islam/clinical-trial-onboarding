from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "eval_runs"
TEX_PATH = OUT / "onboarding_eval_results_table_template.tex"

latex = r"""
\begin{table}[t]
\centering
\caption{Planned multi-case onboarding evaluation summary. Percentages and counts will be filled after structured manual audit.}
\label{tab:onboarding_eval_results}
\begin{tabular}{lc}
\toprule
Metric & Value \\
\midrule
Number of evaluation cases & 15 \\
Eligibility overstatement rate & TBD \\
Explanation grounding acceptable & TBD \\
Fallback used correctly & TBD \\
Missing-fact identification acceptable & TBD \\
Unresolved-requirement identification acceptable & TBD \\
Teach-back targeting acceptable & TBD \\
Overall usable outputs & TBD \\
\botrule
\end{tabular}
\end{table}
""".strip() + "\n"

with open(TEX_PATH, "w") as f:
    f.write(latex)

print("Saved:", TEX_PATH)
