"""
Generate G→V→R reasoning trajectories for Somadhan.

Usage for two-chunk distributed generation:

  Machine 1:  python generate_trajectories.py --id-start 1
  Machine 2:  python generate_trajectories.py --id-start 1001

After both finish, merge the outputs:

  cat data/somadhan_trajectories.jsonl (from machine 1)
      data/somadhan_trajectories.jsonl (from machine 2)
    > data/all_trajectories.jsonl
"""

import argparse
from pathlib import Path

from malt.search import TreeSearchConfig, run_tree_search_for_somadhan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MALT tree search on a Somadhan CSV chunk.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/SOMADHAN.csv",
        help="Path to the Somadhan CSV file (this chunk).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/somadhan_trajectories.jsonl",
        help="Output JSONL path for trajectories.",
    )
    parser.add_argument(
        "--id-start",
        type=int,
        default=1,
        help="First ID to assign to rows in this chunk. "
             "Use 1 for chunk 1, 1001 for chunk 2, etc.",
    )
    parser.add_argument(
        "--branching-factor",
        type=int,
        default=3,
        help="Branching factor n (produces n^3 trajectories per question).",
    )
    args = parser.parse_args()

    cfg = TreeSearchConfig(
        branching_factor=args.branching_factor,
        output_path=Path(args.output_path),
        use_torch_compile=False,
    )

    run_tree_search_for_somadhan(
        csv_path=Path(args.csv_path),
        cfg=cfg,
        id_start=args.id_start,
    )


if __name__ == "__main__":
    main()
