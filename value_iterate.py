"""
Apply value iteration to Somadhan trajectories.

If data/all_trajectories.jsonl already exists (manually merged), uses it
directly.  Otherwise, auto-merges from output/trajectories/ first.

  python value_iterate.py
"""

from pathlib import Path
from malt.search.value_iteration import ValueIterationConfig, value_iteration_over_jsonl

merged_path = Path("data/all_trajectories.jsonl")

if not merged_path.exists():
    from generate_trajectories import merge_jsonl_files
    traj_dir = Path("output/trajectories")
    print(f"Merged file not found. Auto-merging from {traj_dir} ...")
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merge_jsonl_files(traj_dir, merged_path)

value_iteration_over_jsonl(merged_path, cfg=ValueIterationConfig(task="somadhan"))
