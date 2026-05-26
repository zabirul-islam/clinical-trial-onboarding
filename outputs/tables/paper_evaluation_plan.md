## Evaluation plan for the onboarding module

| Category | What to measure | Proposed method |
|---|---|---|
| Retrieval quality | nDCG@10, nDCG@20, Recall@20, Recall@100 | TREC Clinical Trials judged retrieval benchmark |
| Eligibility safety | Overstatement rate, uncertainty calibration, support for missing facts | Manual review of structured eligibility outputs |
| Explanation grounding | Unsupported-field rate, fallback accuracy, passage support consistency | Human audit of explanation JSON fields |
| Patient clarity | Plain-language readability and factual faithfulness | Expert review plus rubric-based scoring |
| Teach-back usefulness | Whether generated questions target real missing or critical concepts | Human evaluation against pipeline state |
| End-to-end utility | Whether the system helps identify what is known, unknown, and still needs confirmation | Scenario-based evaluation on representative onboarding cases |
