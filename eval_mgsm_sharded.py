from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path


def split_tsv_shuffled(
    src_tsv: Path,
    out_dir: Path,
    shard_size: int,
    seed: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = src_tsv.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]

    idxs = list(range(len(lines)))
    random.Random(seed).shuffle(idxs)

    shard_paths: list[Path] = []
    for shard_id, start in enumerate(range(0, len(idxs), shard_size)):
        shard_idxs = idxs[start : start + shard_size]
        shard_lines = [lines[i] for i in shard_idxs]
        shard_path = out_dir / f"mgsm_bn_{shard_id:03d}.tsv"
        shard_path.write_text("\n".join(shard_lines) + "\n", encoding="utf-8")
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

    # Use repo root as cwd so relative paths + PYTHONPATH work naturally.
    env = dict(**{k: v for k, v in dict(**(getattr(__import__("os"), "environ"))).items()})
    # Ensure malt package resolves when script runs.
    env["PYTHONPATH"] = str(repo_root)

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument("--src", type=Path, default=Path("data/mgsm_bn.tsv"))
    ap.add_argument("--eval-script", type=Path, default=Path("eval_with_ans.py"))
    ap.add_argument("--shards-dir", type=Path, default=Path("data/shards/mgsm_bn_100_seed42"))
    ap.add_argument("--out-root", type=Path, default=Path("runs/mgsm_bn_sharded"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=20, help="Eval batch size inside each phase")
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

    print(f"Repo root: {repo_root}")
    print(f"Source:    {src}")
    print(f"Shards:    {shards_dir}")
    print(f"Out root:  {out_root}")
    print(f"Eval:      {eval_script}")
    print()

    shard_paths = split_tsv_shuffled(
        src_tsv=src,
        out_dir=shards_dir,
        shard_size=args.shard_size,
        seed=args.seed,
    )
    print(f"Prepared {len(shard_paths)} shard(s)")

    for shard_path in shard_paths:
        shard_name = shard_path.stem  # e.g. mgsm_bn_000
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
                    gen_ckpt=(repo_root / args.gen_ckpt).resolve(),
                    ver_ckpt=(repo_root / args.ver_ckpt).resolve(),
                    ref_ckpt=(repo_root / args.ref_ckpt).resolve(),
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