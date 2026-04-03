"""
evaluate_ganitllm.py
--------------------
Evaluates the dipta007/GanitLLM-4B_SFT_CGRPO model on a math word-problem CSV.

CSV expected columns:
  (index), query, equation, response, m_query

The script:
  1. Loads the CSV.
  2. For each row, formats `m_query` (Bengali question) using the official
     GanitLLM prompt format with chat template.
  3. Generates a response and extracts the answer from <answer> </answer> tags.
  4. Compares it with the ground-truth `response` column.
  5. Prints a detailed per-sample report and aggregate metrics.

Requirements:
  pip install transformers torch accelerate pandas tqdm

Usage:
  python evaluate_ganitllm.py --csv path/to/data.csv [--max_new_tokens 2048]
                               [--temperature 0.7] [--output_json results.json]
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Model ID ──────────────────────────────────────────────────────────────────
MODEL_ID = "dipta007/GanitLLM-4B_SFT_CGRPO"

# ── Prompt builder (official format) ─────────────────────────────────────────
def build_prompt(question: str) -> str:
    """
    Uses the official GanitLLM prompt:
    Instructs the model to reason step-by-step in Bengali and place the
    final answer inside <answer> </answer> tags.
    """
    return (
        "A conversation takes place between the user and the assistant. "
        "The user asks a question, and the assistant solves the problem. "
        "Please reason step by step in Bengali, and put your final answer "
        "in the <answer> </answer> tags.\n"
        f"Question: {question}"
    )


# ── Answer extraction ─────────────────────────────────────────────────────────
_ANSWER_TAG_RE  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUMBER_RE      = re.compile(r"-?\d+(?:[.,]\d+)?")

def extract_answer(response: str) -> float | None:
    """
    1. Try to pull the value from <answer>...</answer> tags.
    2. Fall back to the first number found anywhere in the response.
    """
    # Primary: answer tag
    m = _ANSWER_TAG_RE.search(response)
    if m:
        candidate = m.group(1).strip()
        num = _parse_number(candidate)
        if num is not None:
            return num

    # Fallback: first number in the full response
    return _parse_number(response)


def _parse_number(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text.replace(",", "").replace("،", ""))
    except ValueError:
        pass
    m = _NUMBER_RE.search(text)
    if m:
        return float(m.group().replace(",", ""))
    return None


def numbers_close(pred: float, gold: float, tol: float = 1e-3) -> bool:
    if gold == 0:
        return abs(pred) < tol
    return abs(pred - gold) / (abs(gold) + 1e-9) < tol


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: Install dependencies with:  pip install transformers torch accelerate")
        sys.exit(1)

    print(f"Loading model  {MODEL_ID}  (this may take a few minutes) …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


# ── Single-sample inference ───────────────────────────────────────────────────
def predict_one(question: str, tokenizer, model, max_new_tokens: int, temperature: float) -> str:
    prompt_text = build_prompt(question)
    messages    = [{"role": "user", "content": prompt_text}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    # Decode only newly generated tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate(csv_path: str, max_new_tokens: int, temperature: float, output_json: str | None):
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, index_col=0)
    required = {"query", "equation", "response", "m_query"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"ERROR: CSV is missing columns: {missing}")
        sys.exit(1)

    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    print(f"Loaded {len(df)} samples from '{csv_path}'\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    tokenizer, model = load_model()

    # ── Inference ──────────────────────────────────────────────────────────────
    all_raw:  list[str]          = []
    all_pred: list[float | None] = []

    try:
        from tqdm import tqdm
        rows = tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="sample")
    except ImportError:
        rows = df.iterrows()

    for _, row in rows:
        raw  = predict_one(str(row["m_query"]), tokenizer, model, max_new_tokens, temperature)
        pred = extract_answer(raw)
        all_raw.append(raw)
        all_pred.append(pred)

    # ── Metrics ────────────────────────────────────────────────────────────────
    records    = []
    n_exact    = 0
    n_parsed   = 0
    abs_errors = []

    print("\n" + "═" * 90)
    print(f"{'#':>4}  {'Gold':>8}  {'Pred':>8}  {'Match':^6}  {'Extracted from <answer> tag / response'}") 
    print("─" * 90)

    for i, (_, row) in enumerate(df.iterrows()):
        gold = float(row["response"]) if pd.notna(row["response"]) else None
        pred = all_pred[i]
        raw  = all_raw[i]

        # Check if answer tag was present
        tag_match = bool(_ANSWER_TAG_RE.search(raw))
        parsed    = pred is not None
        exact     = parsed and gold is not None and numbers_close(pred, gold)

        if parsed:
            n_parsed += 1
        if exact:
            n_exact += 1
        if parsed and gold is not None:
            abs_errors.append(abs(pred - gold))

        match_str = "✓" if exact else ("✗")
        pred_str  = f"{pred:.4g}" if pred is not None else "N/A"
        gold_str  = f"{gold:.4g}" if gold is not None else "N/A"
        tag_note  = "[tag]" if tag_match else "[fallback]"

        # Show last 60 chars of raw (usually contains the answer region)
        preview = raw.strip().replace("\n", " ")[-60:]
        print(f"{i:>4}  {gold_str:>8}  {pred_str:>8}  {match_str:^6}  {tag_note} …{preview}")

        records.append({
            "index":       int(row.name) if pd.notna(row.name) else i,
            "query_en":    str(row["query"]),
            "query_bn":    str(row["m_query"]),
            "equation":    str(row["equation"]),
            "gold":        gold,
            "full_response": raw,
            "answer_tag_found": tag_match,
            "predicted":   pred,
            "correct":     bool(exact),
        })

    total      = len(df)
    accuracy   = n_exact  / total * 100
    parse_rate = n_parsed / total * 100
    tag_rate   = sum(r["answer_tag_found"] for r in records) / total * 100
    mae        = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")

    print("═" * 90)
    print(f"\n📊  EVALUATION REPORT  —  {MODEL_ID}\n")
    print(f"  Total samples          : {total}")
    print(f"  <answer> tag found     : {sum(r['answer_tag_found'] for r in records)}/{total}  ({tag_rate:.1f}%)")
    print(f"  Parse rate             : {n_parsed}/{total}  ({parse_rate:.1f}%)")
    print(f"  Exact-match accuracy   : {n_exact}/{total}  ({accuracy:.1f}%)")
    print(f"  Mean absolute error    : {mae:.4f}" if mae == mae else f"  Mean absolute error    : N/A")
    print()

    # ── Per-operator breakdown ────────────────────────────────────────────────
    op_map: dict[str, dict] = {}
    for rec, (_, row) in zip(records, df.iterrows()):
        op = str(row["equation"]).strip().split()[0]   # first token: +, -, *, /
        op_map.setdefault(op, {"total": 0, "correct": 0})
        op_map[op]["total"]   += 1
        op_map[op]["correct"] += int(rec["correct"])

    if op_map:
        print("  Breakdown by operator:")
        for op, stats in sorted(op_map.items()):
            pct = stats["correct"] / stats["total"] * 100
            print(f"    {op:4s}  {stats['correct']}/{stats['total']}  ({pct:.1f}%)")
        print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    if output_json:
        summary = {
            "model":                MODEL_ID,
            "csv":                  csv_path,
            "max_new_tokens":       max_new_tokens,
            "temperature":          temperature,
            "total":                total,
            "answer_tag_rate_pct":  round(tag_rate,   2),
            "parse_rate_pct":       round(parse_rate, 2),
            "n_correct":            n_exact,
            "accuracy_pct":         round(accuracy,   2),
            "mean_absolute_error":  round(mae, 4) if mae == mae else None,
            "operator_breakdown":   op_map,
            "samples":              records,
        }
        Path(output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"  Full results saved → {output_json}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dipta007/GanitLLM-4B_SFT_CGRPO on a math word-problem CSV."
    )
    parser.add_argument("--csv",            required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--max_new_tokens", type=int,   default=2048,
                        help="Max tokens to generate per sample (default: 2048)")
    parser.add_argument("--temperature",    type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--output_json",    default=None,
                        help="Optional path to save full results as JSON")
    args = parser.parse_args()

    evaluate(
        csv_path       = args.csv,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        output_json    = args.output_json,
    )


if __name__ == "__main__":
    main()