from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

PIPE_PATH = OUT / "onboarding_pipeline_output.json"
CONSENT_PATH = OUT / "consent_explanation_with_unclear.json"
TEACH_PATH = OUT / "teachback_questions_targeted.json"

METHODS_MD = OUT / "paper_methods_onboarding.md"
EXAMPLE_MD = OUT / "paper_pipeline_example.md"
EVAL_MD = OUT / "paper_evaluation_plan.md"


def main():
    with open(PIPE_PATH, "r") as f:
        pipe = json.load(f)

    with open(CONSENT_PATH, "r") as f:
        consent = json.load(f)

    with open(TEACH_PATH, "r") as f:
        teach = json.load(f)

    # Methods subsection
    methods = f"""## Onboarding pipeline

We implemented a retrieval-grounded onboarding pipeline for clinical trial patient interaction. The pipeline accepts a patient-facing question and returns four structured outputs: retrieved evidence passages, an uncertainty-aware eligibility assessment, a postprocessed consent-style explanation, and teach-back questions.

The pipeline operates in five stages. First, a lexical retriever over the full trial text generates candidate trial documents. Second, a cross-encoder reranks candidate evidence and selects top supporting passages. Third, a guarded eligibility module produces an uncertainty-aware decision with supported patient facts, missing patient facts, and unresolved study requirements. Fourth, a strict consent explanation module generates field-specific plain-language explanations, which are then postprocessed with evidence-bounded fallback rules so that unsupported fields are replaced with “not clearly stated in the retrieved evidence.” Fifth, a teach-back module generates patient-facing questions targeted to the actual missing or unresolved concepts identified by the eligibility module.

To reduce unsafe overstatement, the system is prohibited from making definitive eligibility claims unless the retrieved evidence fully supports them. In the current implementation, the eligibility module instead uses conservative categories such as likely match, possible match with insufficient evidence, unlikely match, and cannot determine. This design allows the onboarding agent to support patient understanding while preserving the study team’s role in final eligibility confirmation.
"""

    # Example table
    elig = pipe["eligibility_assessment"]
    example = f"""## Example pipeline output

| Component | Output |
|---|---|
| Input question | {pipe['input_question']} |
| Eligibility decision | {elig['decision']} |
| Patient-facing answer | {elig['patient_facing_answer']} |
| Supported patient facts | {'; '.join(elig.get('supported_patient_facts', []))} |
| Missing patient facts | {'; '.join(elig.get('missing_patient_facts', []))} |
| Unresolved study requirements | {'; '.join(elig.get('unresolved_study_requirements', []))} |
| Study purpose | {consent['study_purpose']['text']} |
| Who may join | {consent['who_may_join']['text']} |
| Procedures or participation | {consent['procedures_or_participation']['text']} |
| Risks or burdens | {consent['risks_or_burdens']['text']} |
| Time commitment | {consent['time_commitment']['text']} |
| What is unclear | {'; '.join(consent.get('what_is_unclear', []))} |
| Rebuilt patient-facing summary | {pipe['consent_explanation']['patient_facing_summary_rebuilt']['text']} |
| Targeted teach-back questions | {'; '.join(teach.get('teachback_questions', []))} |
"""
    # Evaluation plan
    eval_md = """## Evaluation plan for the onboarding module

| Category | What to measure | Proposed method |
|---|---|---|
| Retrieval quality | nDCG@10, nDCG@20, Recall@20, Recall@100 | TREC Clinical Trials judged retrieval benchmark |
| Eligibility safety | Overstatement rate, uncertainty calibration, support for missing facts | Manual review of structured eligibility outputs |
| Explanation grounding | Unsupported-field rate, fallback accuracy, passage support consistency | Human audit of explanation JSON fields |
| Patient clarity | Plain-language readability and factual faithfulness | Expert review plus rubric-based scoring |
| Teach-back usefulness | Whether generated questions target real missing or critical concepts | Human evaluation against pipeline state |
| End-to-end utility | Whether the system helps identify what is known, unknown, and still needs confirmation | Scenario-based evaluation on representative onboarding cases |
"""

    METHODS_MD.write_text(methods)
    EXAMPLE_MD.write_text(example)
    EVAL_MD.write_text(eval_md)

    print("Saved:")
    print(METHODS_MD)
    print(EXAMPLE_MD)
    print(EVAL_MD)


if __name__ == "__main__":
    main()
