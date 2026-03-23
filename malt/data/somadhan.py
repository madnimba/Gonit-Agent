from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _bengali_to_arabic(text: str) -> str:
    """Convert Bengali digits (০-৯) to Arabic digits (0-9)."""
    return text.translate(_BENGALI_DIGITS)


@dataclass
class SomadhanExample:
    """
    Canonical representation of a Somadhan example.

    Attributes:
        id: String identifier for the example.
        question: Problem statement.
        answer_raw: Original solution text from the dataset.
        answer_target: Canonicalized final answer string (for evaluation).
    """

    id: str
    question: str
    answer_raw: str
    answer_target: str


def extract_Somadhan_answer(answer_text: str) -> str:
    """
    Extract the final answer from a Somadhan solution or model output.

    Handles multiple formats in priority order:
      1. <answer>...</answer> tags  (GanitLLM model output)
      2. #### marker               (SOMADHAN CSV ground truth)
      3. Last numeric token         (fallback)

    Any <think>...</think> blocks are stripped before extraction.
    Bengali digits are converted to Arabic in the result.
    """
    # Strip thinking blocks that Qwen3 models sometimes produce.
    text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL)

    # 1) <answer>...</answer> tags
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if m:
        return _bengali_to_arabic(m.group(1).strip())

    # 2) #### marker (GSM8K / SOMADHAN ground-truth style)
    if "####" in text:
        final = text.split("####", maxsplit=1)[-1]
        return _bengali_to_arabic(final.strip())

    # 3) Last numeric token (Bengali or Arabic)
    numeric_matches = re.findall(r"-?[\d০-৯]+(?:\.[\d০-৯]+)?", text)
    if numeric_matches:
        return _bengali_to_arabic(numeric_matches[-1].strip())

    return _bengali_to_arabic(text.strip())


def normalize_Somadhan_answer(text: str) -> str:
    """
    Normalize a Somadhan answer string for comparison.

    Bengali digits are converted to Arabic first, then:
    - Numeric strings are normalized to compact form (no trailing .0).
    - Non-numeric strings are lowercased and stripped.
    """
    text = _bengali_to_arabic(text.strip())
    cleaned = text.replace(",", "")
    try:
        value = float(cleaned)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return cleaned.lower()


def Somadhan_exact_match(predicted: str, target: str) -> bool:
    """
    Exact-match comparator for Somadhan answers.

    Both strings are passed through extract + normalize before comparison.
    """
    pred_final = normalize_Somadhan_answer(extract_Somadhan_answer(predicted))
    target_final = normalize_Somadhan_answer(extract_Somadhan_answer(target))
    return pred_final == target_final


def load_Somadhan_split(
    csv_path: str | Path,
    id_start: int = 1,
) -> List[SomadhanExample]:
    """
    Load the full Somadhan dataset from a CSV file.

    The CSV is expected to have at minimum the columns:
        - question
        - answer

    An optional 'id' column is used if present; otherwise IDs are assigned
    sequentially starting from *id_start*.  For chunked trajectory
    generation across multiple machines, use different id_start values
    (e.g. 1 for chunk 1, 1001 for chunk 2) so that IDs are globally unique
    after merging the output JSONL files.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    examples: List[SomadhanExample] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {csv_path}")

        missing = {"question", "answer"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {list(reader.fieldnames)}"
            )

        for idx, row in enumerate(reader):
            question = row.get("question", "").strip()
            answer_raw = row.get("answer", "").strip()
            answer_target = extract_Somadhan_answer(answer_raw)
            examples.append(
                SomadhanExample(
                    id=str(row.get("id", id_start + idx)),
                    question=question,
                    answer_raw=answer_raw,
                    answer_target=answer_target,
                )
            )

    return examples
