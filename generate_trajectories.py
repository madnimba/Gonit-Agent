"""
Generate G→V→R reasoning trajectories for Somadhan.

Loads the published GanitLLM checkpoint from Hugging Face with no LoRA adapters
(same weights for generator, verifier, and refiner prompts).

Outputs are split into JSONL files of --chunk-size samples each (default
500), written to an organized output folder.  A log file tracks progress
so that a crashed run can be resumed cleanly.

Usage for two-machine distributed generation:

  Machine 1:  python generate_trajectories.py --id-start 1
  Machine 2:  python generate_trajectories.py --id-start 1001

Output structure (example with id-start 1, chunk-size 500):

  output/trajectories/
    trajectories_0001_0500.jsonl
    trajectories_0501_1000.jsonl
    ...
  output/generation_log.txt

After both machines finish, merge all JSONL files:

  python generate_trajectories.py --merge-only --merge-dir output/trajectories
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _log_event(log_path: Path, message: str) -> None:
    """Append a timestamped line to the log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def _get_last_completed_id(log_path: Path) -> int | None:
    """Parse the log file to find the last successfully completed sample ID."""
    if not log_path.exists():
        return None
    last_id = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "COMPLETED sample id=" in line:
                try:
                    last_id = int(line.split("id=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
    return last_id


def merge_jsonl_files(trajectories_dir: Path, output_path: Path) -> None:
    """
    Merge all JSONL files in *trajectories_dir* into a single file at
    *output_path*, sorted by filename (which encodes ID order).
    """
    jsonl_files = sorted(trajectories_dir.glob("trajectories_*.jsonl"))
    if not jsonl_files:
        log.warning("No trajectory files found in %s", trajectories_dir)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8") as out:
        for jf in jsonl_files:
            with jf.open("r", encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        total += 1

    log.info("Merged %d trajectories from %d files into %s",
             total, len(jsonl_files), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MALT tree search on a Somadhan CSV chunk.",
    )
    parser.add_argument(
        "--csv-path", type=str, default="data/SOMADHAN.csv",
        help="Path to the Somadhan CSV file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Root output directory. Trajectories go into output/trajectories/.",
    )
    parser.add_argument(
        "--id-start", type=int, default=1,
        help="First ID for rows in this chunk (1 for chunk 1, 1001 for chunk 2, etc.).",
    )
    parser.add_argument(
        "--branching-factor", type=int, default=3,
        help="Branching factor n (produces n^3 trajectories per question).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Number of samples per output JSONL file.",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip generation; just merge existing JSONL files.",
    )
    parser.add_argument(
        "--merge-dir", type=str, default=None,
        help="Directory of JSONL files to merge (default: <output-dir>/trajectories).",
    )
    parser.add_argument(
        "--merge-output", type=str, default="data/all_trajectories.jsonl",
        help="Path for the merged output file.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    traj_dir = output_root / "trajectories"
    log_path = output_root / "generation_log.txt"

    # ------------------------------------------------------------------
    # Merge-only mode
    # ------------------------------------------------------------------
    if args.merge_only:
        src = Path(args.merge_dir) if args.merge_dir else traj_dir
        merge_jsonl_files(src, Path(args.merge_output))
        return

    # ------------------------------------------------------------------
    # Generation mode
    # ------------------------------------------------------------------
    from malt.data import load_Somadhan_split
    from malt.models import MaltModelConfig, load_ganit_llm_base
    from malt.search.tree_search import (
        TreeSearchConfig,
        run_tree_search_for_questions,
    )

    traj_dir.mkdir(parents=True, exist_ok=True)

    examples = load_Somadhan_split(args.csv_path, id_start=args.id_start)
    log.info("Loaded %d examples (id_start=%d)", len(examples), args.id_start)

    # Resume: skip already-completed samples
    last_done_id = _get_last_completed_id(log_path)
    if last_done_id is not None:
        before = len(examples)
        examples = [ex for ex in examples if int(ex.id) > last_done_id]
        log.info("Resuming after id %d — skipping %d already-processed samples",
                 last_done_id, before - len(examples))
        _log_event(log_path, f"RESUMED after id={last_done_id}, {len(examples)} remaining")

    if not examples:
        log.info("Nothing to process.")
        return

    cfg = TreeSearchConfig(
        branching_factor=args.branching_factor,
        use_torch_compile=False,
    )

    model, tokenizer = load_ganit_llm_base(MaltModelConfig())
    log.info("Model loaded (GanitLLM HF base, no LoRA)")

    # ------------------------------------------------------------------
    # Process in chunks of --chunk-size, each chunk → one JSONL file
    # ------------------------------------------------------------------
    chunk_size = args.chunk_size
    total = len(examples)
    times: List[float] = []

    chunk_first_id: str | None = None
    chunk_last_id: str | None = None
    samples_in_chunk = 0
    current_chunk_path: Path | None = None
    current_chunk_fh = None  # open file handle for the active chunk

    def _start_new_chunk(first_id: str) -> None:
        """Open a new chunk file for writing."""
        nonlocal chunk_first_id, chunk_last_id, samples_in_chunk
        nonlocal current_chunk_path, current_chunk_fh
        chunk_first_id = first_id
        chunk_last_id = first_id
        samples_in_chunk = 0
        # Temporary name while the chunk is being written; renamed on close.
        current_chunk_path = traj_dir / f"trajectories_{int(first_id):06d}_partial.jsonl"
        current_chunk_fh = current_chunk_path.open("w", encoding="utf-8")
        _log_event(log_path, f"STARTING chunk — first sample id={first_id}")

    def _close_chunk() -> None:
        """Close the current chunk file and rename to its final name."""
        nonlocal chunk_first_id, chunk_last_id, samples_in_chunk
        nonlocal current_chunk_path, current_chunk_fh
        if current_chunk_fh is None:
            return
        current_chunk_fh.close()
        current_chunk_fh = None
        final_name = f"trajectories_{int(chunk_first_id):06d}_{int(chunk_last_id):06d}.jsonl"
        final_path = traj_dir / final_name
        current_chunk_path.rename(final_path)
        _log_event(log_path,
                   f"WRITTEN {final_name} — samples id={chunk_first_id} to id={chunk_last_id} "
                   f"({samples_in_chunk} samples)")
        log.info("Wrote %s (%d samples)", final_name, samples_in_chunk)
        current_chunk_path = None
        chunk_first_id = None
        chunk_last_id = None
        samples_in_chunk = 0

    # On resume, a *_partial.jsonl file may be left from the crashed run.
    # The log-based resume already skips completed IDs, so any partial file
    # contains only data for IDs that will be skipped. Remove it.
    for stale in traj_dir.glob("*_partial.jsonl"):
        stale.unlink()
        log.info("Removed stale partial file: %s", stale.name)

    _log_event(log_path,
               f"STARTED generation — {total} samples, chunk_size={chunk_size}, "
               f"branching={args.branching_factor}")

    for idx, (traj, elapsed) in enumerate(
        run_tree_search_for_questions(
            model=model,
            tokenizer=tokenizer,
            questions=examples,
            cfg=cfg,
        )
    ):
        sample_id = traj["id"]

        # Open a new chunk file if needed.
        if current_chunk_fh is None:
            _start_new_chunk(sample_id)

        # Write trajectory to disk immediately — survives crashes.
        current_chunk_fh.write(json.dumps(traj, ensure_ascii=False) + "\n")
        current_chunk_fh.flush()

        chunk_last_id = sample_id
        samples_in_chunk += 1

        _log_event(log_path, f"COMPLETED sample id={sample_id}")

        # Close chunk when full and start a fresh one on next iteration.
        if samples_in_chunk >= chunk_size:
            _close_chunk()

        # ETA logging
        times.append(elapsed)
        processed = idx + 1
        remaining = total - processed
        avg = sum(times) / len(times)

        if remaining > 0 and processed % 10 == 0:
            eta_s = avg * remaining
            h, r = divmod(int(eta_s), 3600)
            m, s = divmod(r, 60)
            eta_str = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")
            log.info(
                "Progress: %d/%d (%.1f%%) | Last: %.1fs | Avg: %.1fs | ETA: %s",
                processed, total, 100.0 * processed / total,
                elapsed, avg, eta_str,
            )

    # Close the last (possibly partial) chunk.
    _close_chunk()

    _log_event(log_path, f"FINISHED — all {total} samples processed")
    log.info("Done. Trajectories in %s", traj_dir)


if __name__ == "__main__":
    main()
