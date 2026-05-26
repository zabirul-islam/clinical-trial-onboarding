from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

bm25 = pd.read_csv(OUT / "bm25_variants_summary_metrics.csv")
dense_minilm_elig = pd.read_csv(OUT / "dense_summary_metrics.csv")
dense_minilm_all = pd.read_csv(OUT / "dense_summary_metrics__sentence_transformers__all_MiniLM_L6_v2__text_all.csv")
dense_bge_all = pd.read_csv(OUT / "dense_summary_metrics__BAAI__bge_base_en_v1.5__text_all.csv")
cross = pd.read_csv(OUT / "crossenc_summary_metrics.csv")

rows = []

# BM25 full-text
r = bm25[bm25["run_name"] == "bm25_all_text"].iloc[0]
rows.append({
    "method": "BM25 (full trial text)",
    "stage": "single-stage lexical",
    "nDCG@10": r["nDCG@10"],
    "nDCG@20": r["nDCG@20"],
    "Recall@10": r["Recall@10"],
    "Recall@20": r["Recall@20"],
    "Recall@100": r["Recall@100"],
})

# MiniLM eligibility-focus
r = dense_minilm_elig.iloc[0]
rows.append({
    "method": "Dense MiniLM (eligibility-focused text)",
    "stage": "single-stage dense",
    "nDCG@10": r["nDCG@10"],
    "nDCG@20": r["nDCG@20"],
    "Recall@10": r["Recall@10"],
    "Recall@20": r["Recall@20"],
    "Recall@100": r["Recall@100"],
})

# MiniLM full text
r = dense_minilm_all.iloc[0]
rows.append({
    "method": "Dense MiniLM (full trial text)",
    "stage": "single-stage dense",
    "nDCG@10": r["nDCG@10"],
    "nDCG@20": r["nDCG@20"],
    "Recall@10": r["Recall@10"],
    "Recall@20": r["Recall@20"],
    "Recall@100": r["Recall@100"],
})

# BGE full text
r = dense_bge_all.iloc[0]
rows.append({
    "method": "Dense BGE-base (full trial text)",
    "stage": "single-stage dense",
    "nDCG@10": r["nDCG@10"],
    "nDCG@20": r["nDCG@20"],
    "Recall@10": r["Recall@10"],
    "Recall@20": r["Recall@20"],
    "Recall@100": r["Recall@100"],
})

# Cross-encoder rerank
r = cross.iloc[0]
rows.append({
    "method": "BM25 full-text → Cross-encoder rerank",
    "stage": "two-stage",
    "nDCG@10": r["nDCG@10"],
    "nDCG@20": r["nDCG@20"],
    "Recall@10": r["Recall@10"],
    "Recall@20": r["Recall@20"],
    "Recall@100": r["Recall@100"],
})

df = pd.DataFrame(rows)

# save csv
df.to_csv(OUT / "final_retrieval_comparison.csv", index=False)

# pretty csv
pretty = df.copy()
for col in ["nDCG@10", "nDCG@20", "Recall@10", "Recall@20", "Recall@100"]:
    pretty[col] = pretty[col].map(lambda x: f"{x:.4f}")
pretty.to_csv(OUT / "final_retrieval_comparison_pretty.csv", index=False)

# latex
latex = []
latex.append("\\begin{table}[t]")
latex.append("\\centering")
latex.append("\\caption{Retrieval comparison on TREC Clinical Trials 2021.}")
latex.append("\\label{tab:retrieval_final}")
latex.append("\\begin{tabular}{lccccc}")
latex.append("\\hline")
latex.append("Method & nDCG@10 & nDCG@20 & R@10 & R@20 & R@100 \\\\")
latex.append("\\hline")

for _, row in pretty.iterrows():
    latex.append(
        f"{row['method']} & {row['nDCG@10']} & {row['nDCG@20']} & "
        f"{row['Recall@10']} & {row['Recall@20']} & {row['Recall@100']} \\\\"
    )

latex.append("\\hline")
latex.append("\\end{tabular}")
latex.append("\\end{table}")

with open(OUT / "final_retrieval_comparison.tex", "w") as f:
    f.write("\n".join(latex))

print("Saved:")
print(OUT / "final_retrieval_comparison.csv")
print(OUT / "final_retrieval_comparison_pretty.csv")
print(OUT / "final_retrieval_comparison.tex")
print()
print(pretty.to_string(index=False))
