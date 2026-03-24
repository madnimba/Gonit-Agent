from pathlib import Path
from malt.search import TreeSearchConfig, run_tree_search_for_somadhan

dataset_csv_path=Path("data/SOMADHAN.csv")
trajectory_output_path = Path("data/somadhan_trajectories_1024.jsonl")

cfg = TreeSearchConfig(branching_factor=3, 
                       output_path=trajectory_output_path,
                       use_torch_compile=False)

run_tree_search_for_somadhan(csv_path=dataset_csv_path, 
                             cfg=cfg)