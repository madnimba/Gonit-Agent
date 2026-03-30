from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def split_csv_shuffled(
    src_csv: Path,
    out_dir: Path,
    shard_size: int,
    seed: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    with src_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {src_csv}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    rng = random.Random(seed)
    rng.shuffle(rows)

    shard_paths: list[Path] = []
    for shard_id, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        shard_path = out_dir / f"bn_msvamp_{shard_id:03d}.csv"
        with shard_path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shard_rows)
        shard_paths.append(shard_path)

    return shard_paths


def run_one_shard(
    repo_root: Path,
    eval_script: Path,
    shard_path: Path,
    shard_out_dir: Path,
    *,
    chunk_size: int,
    num_samples: int,
    gen_ckpt: Path,
    ver_ckpt: Path,
    ref_ckpt: Path,
    quiet: bool,
) -> None:
    shard_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(eval_script),
        "--devset",
        str(shard_path),
        "--output-dir",
        str(shard_out_dir),
        "--chunk-size",
        str(chunk_size),
        "--num-samples",
        str(num_samples),
        "--gen-checkpoint",
        str(gen_ckpt),
        "--ver-checkpoint",
        str(ver_ckpt),
        "--ref-checkpoint",
        str(ref_ckpt),
        "--resume",
    ]
    if quiet:
        cmd.append("--quiet")

    env = os.environ.copy()
    # Ensure `malt` package is importable when running from repo root.
    env["PYTHONPATH"] = str(repo_root)

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument("--src", type=Path, default=Path("data/bn-msvamp.csv"))
    ap.add_argument("--eval-script", type=Path, default=Path("eval_with_ans.py"))

    ap.add_argument("--shards-dir", type=Path, default=Path("data/shards/bn_msvamp_100_seed42"))
    ap.add_argument("--out-root", type=Path, default=Path("runs/bn_msvamp_sharded"))

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-size", type=int, default=100)

    ap.add_argument("--chunk-size", type=int, default=20, help="Batch size inside eval phases")
    ap.add_argument("--num-samples", type=int, default=3, help="MV@k for phase 2 only")

    ap.add_argument("--gen-ckpt", type=Path, required=True)
    ap.add_argument("--ver-ckpt", type=Path, required=True)
    ap.add_argument("--ref-ckpt", type=Path, required=True)

    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    src = (repo_root / args.src).resolve()
    eval_script = (repo_root / args.eval_script).resolve()

    shards_dir = (repo_root / args.shards_dir).resolve()
    out_root = (repo_root / args.out_root).resolve()

    gen_ckpt = (repo_root / args.gen_ckpt).resolve()
    ver_ckpt = (repo_root / args.ver_ckpt).resolve()
    ref_ckpt = (repo_root / args.ref_ckpt).resolve()

    print(f"Repo root: {repo_root}")
    print(f"Source:    {src}")
    print(f"Eval:      {eval_script}")
    print(f"Shards:    {shards_dir}")
    print(f"Out root:  {out_root}")
    print(f"Seed:      {args.seed}")
    print(f"Shard size:{args.shard_size}")
    print(f"Chunk size:{args.chunk_size}")
    print(f"MV@k:      {args.num_samples}")
    print()

    shard_paths = split_csv_shuffled(
        src_csv=src,
        out_dir=shards_dir,
        shard_size=args.shard_size,
        seed=args.seed,
    )
    print(f"Prepared {len(shard_paths)} shard(s)")

    for shard_path in shard_paths:
        shard_name = shard_path.stem  # bn_msvamp_000
        shard_out_dir = out_root / shard_name

        for attempt in range(1, args.max_retries + 1):
            try:
                print(f"\n=== Shard {shard_name} attempt {attempt}/{args.max_retries} ===")
                run_one_shard(
                    repo_root=repo_root,
                    eval_script=eval_script,
                    shard_path=shard_path,
                    shard_out_dir=shard_out_dir,
                    chunk_size=args.chunk_size,
                    num_samples=args.num_samples,
                    gen_ckpt=gen_ckpt,
                    ver_ckpt=ver_ckpt,
                    ref_ckpt=ref_ckpt,
                    quiet=args.quiet,
                )
                print(f"Shard {shard_name} DONE. Outputs in {shard_out_dir}")
                break
            except subprocess.CalledProcessError as e:
                print(f"Shard {shard_name} FAILED (exit {e.returncode}).")
                if attempt >= args.max_retries:
                    raise
                print(f"Sleeping {args.retry_sleep_seconds}s then retrying with --resume...")
                time.sleep(args.retry_sleep_seconds)

    print("\nAll shards complete.")


if __name__ == "__main__":
    main()