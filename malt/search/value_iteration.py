from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import json

from malt.data import (
    extract_gsm8k_answer,
    gsm8k_exact_match,
    extract_math_answer,
    math_exact_match,
)
from malt.data import (
    extract_Somadhan_answer,
    Somadhan_exact_match,
)


TaskName = Literal["gsm8k", "math", "somadhan"]


@dataclass
class ValueIterationConfig:
    """
    Configuration for value-iteration-based credit assignment.
    """

    task: TaskName = "somadhan"
    threshold: float = 0.5


def _extract_answer(task: TaskName, text: str) -> str:
    """Task-specific answer extraction function T(ro)."""
    if task == "gsm8k":
        return extract_gsm8k_answer(text)
    if task == "math":
        return extract_math_answer(text)
    if task == "somadhan":
        return extract_Somadhan_answer(text)
    return text.strip()


def _reward(task: TaskName, pred: str, target: str) -> int:
    """Binary reward: 1 if pred matches target under the task's comparator."""
    if task == "gsm8k":
        return 1 if gsm8k_exact_match(pred, target) else 0
    if task == "math":
        return 1 if math_exact_match(pred, target) else 0
    if task == "somadhan":
        return 1 if Somadhan_exact_match(pred, target) else 0
    return 1 if pred.strip() == target.strip() else 0


def compute_values_for_trajectory(
    traj: Dict,
    cfg: ValueIterationConfig,
) -> Dict:
    """
    Compute V(r), V(v), V(g) and binary labels for a single trajectory in-place.

    The trajectory is expected to have the structure produced by
    `run_tree_search_for_questions` in `tree_search.py`.

    We add the following keys:
      - For each refiner node:
          - "answer_pred": extracted final answer string T(ro)
          - "value": 0 or 1 (binary reward)
      - For each verifier node:
          - "value": mean of child refiner values in [0, 1]
          - "label": 1 if value > threshold else 0
      - For each generator node:
          - "value": mean of child verifier values in [0, 1]
          - "label": 1 if value > threshold else 0
    """
    task = cfg.task
    threshold = cfg.threshold
    answer_gt = traj.get("answer_gt", "")

    gen_nodes = traj.get("generator_nodes", [])

    # Leaf values: refiner nodes
    for g in gen_nodes:
        for v in g.get("verifier_nodes", []):
            for r in v.get("refiner_nodes", []):
                text = r.get("text", "")
                pred_ans = _extract_answer(task, text)
                r["answer_pred"] = pred_ans
                r["value"] = _reward(task, pred_ans, answer_gt)

    # Verifier values: mean over refiners
    for g in gen_nodes:
        for v in g.get("verifier_nodes", []):
            ref_nodes = v.get("refiner_nodes", [])
            if not ref_nodes:
                v_value = 0.0
            else:
                total = sum(int(r.get("value", 0)) for r in ref_nodes)
                v_value = total / float(len(ref_nodes))
            v["value"] = v_value
            v["label"] = 1 if v_value > threshold else 0

    # Generator values: mean over verifiers
    for g in gen_nodes:
        ver_nodes = g.get("verifier_nodes", [])
        if not ver_nodes:
            g_value = 0.0
        else:
            total = sum(float(v.get("value", 0.0)) for v in ver_nodes)
            g_value = total / float(len(ver_nodes))
        g["value"] = g_value
        g["label"] = 1 if g_value > threshold else 0

    return traj


def apply_value_iteration_to_trajectories(
    trajectories: Iterable[Dict],
    cfg: ValueIterationConfig,
) -> List[Dict]:
    """
    Apply value iteration to a sequence of trajectories and return a list
    with augmented entries.
    """
    return [compute_values_for_trajectory(traj, cfg) for traj in trajectories]


def value_iteration_over_jsonl(
    input_path: Path,
    output_path: Path | None = None,
    cfg: ValueIterationConfig | None = None,
) -> Path:
    """
    Convenience function: read trajectories from a JSONL file, apply value
    iteration, and write the augmented trajectories back out.
    """
    cfg = cfg or ValueIterationConfig()
    if output_path is None:
        output_path = input_path.with_suffix(".valued.jsonl")

    augmented: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            traj = json.loads(line)
            augmented.append(compute_values_for_trajectory(traj, cfg))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for traj in augmented:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    return output_path