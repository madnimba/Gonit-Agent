from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from itertools import chain
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
    # 1) Check for <answer> tag BEFORE stripping <think> blocks,
    #    because Qwen3 often places <answer> inside <think>.
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", answer_text, flags=re.DOTALL)
    if m:
        return _bengali_to_arabic(m.group(1).strip())

    # Now safe to strip <think> blocks (no <answer> tag was inside them).
    text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL)

    # 2) #### marker
    if "####" in text:
        final = text.split("####", maxsplit=1)[-1]
        return _bengali_to_arabic(final.strip())

    # 3) Last numeric token fallback
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


def _load_tsv_somadhan_split(tsv_path: Path, id_start: int) -> List[SomadhanExample]:
    """Two tab-separated columns per row: question, answer (optional header row)."""
    examples: List[SomadhanExample] = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows_iter = iter(reader)
        first = next(rows_iter, None)
        if first is None:
            return examples
        if (
            len(first) >= 2
            and first[0].strip().lower() == "question"
            and first[1].strip().lower() == "answer"
        ):
            data_rows = rows_iter
        else:
            data_rows = chain([first], rows_iter)

        for idx, row in enumerate(data_rows):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"Row {idx + 1} in {tsv_path}: expected 2 tab-separated columns, got {len(row)}"
                )
            question = row[0].strip()
            answer_raw = row[1].strip()
            answer_target = extract_Somadhan_answer(answer_raw)
            examples.append(
                SomadhanExample(
                    id=str(id_start + len(examples)),
                    question=question,
                    answer_raw=answer_raw,
                    answer_target=answer_target,
                )
            )

    return examples


def load_Somadhan_split(
    csv_path: str | Path,
    id_start: int = 1,
) -> List[SomadhanExample]:
    """
    Load examples from a Somadhan-style dev file.

    Supported formats:

    * **CSV** with columns ``question`` and ``answer`` (optional ``id``), e.g. SOMADHAN.csv.
    * **CSV** with columns ``m_query`` and ``response`` (e.g. bn-msvamp.csv).
    * **TSV** (``.tsv``): two tab-separated columns per line, question then answer.
      An optional header row ``question<TAB>answer`` is recognized; otherwise the
      first line is treated as data (e.g. mgsm_bn.tsv).

    An optional ``id`` column is used when present on CSV rows; otherwise IDs are
    assigned sequentially starting from *id_start*.  For chunked trajectory
    generation across multiple machines, use different id_start values
    (e.g. 1 for chunk 1, 1001 for chunk 2) so that IDs are globally unique
    after merging the output JSONL files.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if csv_path.suffix.lower() == ".tsv":
        return _load_tsv_somadhan_split(csv_path, id_start)

    examples: List[SomadhanExample] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {csv_path}")

        fields = set(reader.fieldnames)
        if "question" in fields and "answer" in fields:
            q_key, a_key = "question", "answer"
        elif "m_query" in fields and "response" in fields:
            q_key, a_key = "m_query", "response"
        else:
            raise ValueError(
                "CSV must have columns (question, answer) or (m_query, response). "
                f"Found columns: {list(reader.fieldnames)}"
            )

        for idx, row in enumerate(reader):
            question = row.get(q_key, "").strip()
            answer_raw = row.get(a_key, "").strip()
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
