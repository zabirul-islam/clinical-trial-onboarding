from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "outputs" / "tables" / "bm25_variants_summary_metrics.csv"
OUTDIR = ROOT / "outputs" / "tables"

df = pd.read_csv(INFILE)

order = [
    "bm25_eligibility_focus",
    "bm25_all_text",
    "rrf_bm25_all_plus_eligibility",
]
df["order"] = df["run_name"].map({name: i for i, name in enumerate(order)})
df = df.sort_values("order").drop(columns=["order"])

pretty = df.copy()
for col in ["nDCG@10", "nDCG@20", "Recall@10", "Recall@20", "Recall@100"]:
    pretty[col] = pretty[col].map(lambda x: f"{x:.4f}")

pretty.to_csv(OUTDIR / "retrieval_results_table.csv", index=False)

latex = []
latex.append("\\begin{table}[t]")
latex.append("\\centering")
latex.append("\\caption{Lexical retrieval baselines on TREC Clinical Trials 2021.}")
latex.append("\\label{tab:retrieval_lexical}")
latex.append("\\begin{tabular}{lccccc}")
latex.append("\\hline")
latex.append("Method & nDCG@10 & nDCG@20 & R@10 & R@20 & R@100 \\\\")
latex.append("\\hline")

name_map = {
    "bm25_eligibility_focus": "BM25 (eligibility-focused text)",
    "bm25_all_text": "BM25 (full trial text)",
    "rrf_bm25_all_plus_eligibility": "RRF fusion of BM25 runs",
}

for _, row in pretty.iterrows():
    latex.append(
        f"{name_map.get(row['run_name'], row['run_name'])} & "
        f"{row['nDCG@10']} & {row['nDCG@20']} & {row['Recall@10']} & {row['Recall@20']} & {row['Recall@100']} \\\\"
    )

latex.append("\\hline")
latex.append("\\end{tabular}")
latex.append("\\end{table}")

with open(OUTDIR / "retrieval_results_table.tex", "w") as f:
    f.write("\n".join(latex))

print("Saved:")
print(OUTDIR / "retrieval_results_table.csv")
print(OUTDIR / "retrieval_results_table.tex")
