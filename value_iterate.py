"""
Apply value iteration to merged Somadhan trajectories.

Run after merging trajectory JSONL files from all machines:

  python value_iterate.py
"""

from pathlib import Path
from malt.search.value_iteration import ValueIterationConfig, value_iteration_over_jsonl

input_path = Path("data/all_trajectories.jsonl")
value_iteration_over_jsonl(input_path, cfg=ValueIterationConfig(task="somadhan"))
