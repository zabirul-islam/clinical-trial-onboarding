from pathlib import Path
import argparse
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

SYSTEM_PROMPT = """You are a clinical trial onboarding assistant.
You must explain trial information using ONLY the provided evidence passages.

Critical grounding rules:
1. Do not invent any study details.
2. Every populated field must be directly supported by at least one evidence passage.
3. If a field is not explicitly supported, write exactly: "not clearly stated in the retrieved evidence"
4. Do not infer risks, burdens, medications, follow-up, or time commitment unless explicitly stated.
5. Do not give medical advice.
6. Use plain, patient-friendly language.
7. Output valid JSON only.
8. Never say things like "might experience side effects" unless the evidence explicitly says so.
"""

USER_TEMPLATE = """Patient request:
{question}

Evidence passages:
{evidence}

Return JSON with exactly this structure:
{{
  "study_purpose": {{
    "text": "...",
    "support_passages": [1, 2]
  }},
  "who_may_join": {{
    "text": "...",
    "support_passages": [1, 2]
  }},
  "procedures_or_participation": {{
    "text": "...",
    "support_passages": [1, 2]
  }},
  "risks_or_burdens": {{
    "text": "...",
    "support_passages": [1, 2]
  }},
  "time_commitment": {{
    "text": "...",
    "support_passages": [1, 2]
  }},
  "what_is_unclear": [
    "...",
    "..."
  ],
  "patient_facing_summary": {{
    "text": "...",
    "support_passages": [1, 2]
  }}
}}

Rules:
- Use only passage_rank numbers that appear in the evidence.
- If text is "not clearly stated in the retrieved evidence", support_passages must be [].
- Keep text short and factual.
- Do not output markdown.
"""

REQUIRED_FALLBACK = "not clearly stated in the retrieved evidence"


def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model, device


def extract_first_json_object(text: str):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=520)
    args = parser.parse_args()

    ev = pd.read_csv(OUT / "grounding_evidence_top_passages.csv")
    evidence_blocks = []
    for _, row in ev.iterrows():
        evidence_blocks.append(
            f"[Passage {int(row['passage_rank'])} | doc={row['doc_id']} | section={row['section']}]\n{row['passage_text']}"
        )
    evidence_text = "\n\n".join(evidence_blocks)

    prompt = USER_TEMPLATE.format(question=args.question, evidence=evidence_text)

    tokenizer, model, device = load_model(args.model_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = SYSTEM_PROMPT + "\n\n" + prompt

    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    raw_path = OUT / "consent_explanation_strict_raw.txt"
    json_path = OUT / "consent_explanation_strict.json"

    with open(raw_path, "w") as f:
        f.write(gen)

    parsed = extract_first_json_object(gen)
    if parsed is not None:
        with open(json_path, "w") as f:
            json.dump(parsed, f, indent=2)

    print("Generated output:\n")
    print(gen)
    print("\nSaved:", raw_path)
    if parsed is not None:
        print("Saved:", json_path)
    else:
        print("Warning: valid JSON was not parsed automatically.")


if __name__ == "__main__":
    main()
