## System architecture figure specification

### Title
Grounded Onboarding Pipeline for Clinical Trial Patient Interaction

### Layout
Use a left-to-right pipeline with 6 main blocks and 2 output branches.

### Blocks
1. **Patient Input**
   - Patient-facing question
   - Optional patient profile / history

2. **Trial Retrieval**
   - BM25 over full trial text
   - Top-K candidate trial documents

3. **Evidence Reranking**
   - Cross-encoder passage reranking
   - Top evidence passages

4. **Eligibility Triage Module**
   - Outputs:
     - decision category
     - supported patient facts
     - missing patient facts
     - unresolved study requirements

5. **Strict Consent Explanation Module**
   - Field-specific explanation
   - Evidence support tracking
   - Postprocessing fallback to “not clearly stated in the retrieved evidence”

6. **Teach-Back Module**
   - Targeted teach-back questions
   - Concepts checked

### Output branches
A. **Patient-facing outputs**
   - cautious eligibility answer
   - plain-language explanation
   - teach-back questions

B. **Audit / safety outputs**
   - retrieved evidence passages
   - support passage IDs
   - missing facts
   - unresolved requirements

### Footer note
Final eligibility remains subject to study-team confirmation.
