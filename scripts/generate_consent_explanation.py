from pathlib import Path
import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

SYSTEM_PROMPT = """You are a clinical trial onboarding assistant.
You explain clinical trial information in plain language using ONLY the provided evidence passages.

Rules:
1. Do not invent any study details.
2. If a detail is not in the evidence, say it is not clearly stated.
3. Do not give medical advice.
4. Use short, patient-friendly language.
5. Be calm, clear, and factual.

You must output valid JSON only.
"""

USER_TEMPLATE = """Patient request:
{question}

Evidence passages:
{evidence}

Return JSON with exactly these keys:
{{
  "study_purpose": "...",
  "who_may_join": "...",
  "procedures_or_participation": "...",
  "risks_or_burdens": "...",
  "time_commitment": "...",
  "what_is_unclear": ["...", "..."],
  "patient_facing_summary": "..."
}}
"""


def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=360)
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

    out_txt = OUT / "consent_explanation_raw.txt"
    out_json = OUT / "consent_explanation.json"

    with open(out_txt, "w") as f:
        f.write(gen)

    start = gen.find("{")
    end = gen.rfind("}")
    parsed = None
    if start != -1 and end != -1 and end > start:
        candidate = gen[start:end + 1]
        try:
            parsed = json.loads(candidate)
            with open(out_json, "w") as f:
                json.dump(parsed, f, indent=2)
        except Exception:
            parsed = None

    print("Generated output:\n")
    print(gen)
    print("\nSaved:", out_txt)
    if parsed is not None:
        print("Saved:", out_json)
    else:
        print("Warning: valid JSON was not parsed automatically.")


if __name__ == "__main__":
    main()
