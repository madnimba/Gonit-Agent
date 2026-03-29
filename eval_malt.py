"""
Evaluate MALT-enhanced GanitLLM vs baseline on Somadhan devset.

Reports four configurations:
  1. GanitLLM base zero-shot (single pass, T=0.0)
  2. GanitLLM base MV@3 (majority vote, T=0.3)
  3. Single-agent generator-only with trained adapter (MV@3)
  4. Multi-agent MALT  G→V→R with trained adapters (MV@3)

Usage (from repo root; ensure ``malt`` is importable, e.g. ``PYTHONPATH=.``):

  PYTHONPATH=. python eval_malt.py --devset data/somadhan_dev.csv

  # With trained checkpoints:
  PYTHONPATH=. python eval_malt.py --devset data/somadhan_dev.csv \\
      --gen-checkpoint checkpoints/generator_sft \\
      --ver-checkpoint checkpoints/verifier_dpo \\
      --ref-checkpoint checkpoints/refiner_dpo

  # Suppress progress lines:
  PYTHONPATH=. python eval_malt.py --devset data/dev.csv --quiet
"""

import argparse
from pathlib import Path

from malt.data import load_Somadhan_split
from malt.inference.pipeline import (
    InferenceConfig,
    run_single_agent_generator,
    run_multi_agent_malt,
)
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_adapters,
    load_malt_llama_with_trained_adapters,
    set_active_role_adapter,
    ROLE_GENERATOR,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.utils.eval import evaluate_somadhan_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MALT on Somadhan devset.")
    parser.add_argument("--devset", type=str, required=True, help="Path to Somadhan dev CSV.")
    parser.add_argument("--num-samples", type=int, default=3, help="MV@k samples.")
    parser.add_argument(
        "--gen-checkpoint", type=str, default=None,
        help="Generator adapter checkpoint directory.",
    )
    parser.add_argument(
        "--ver-checkpoint", type=str, default=None,
        help="Verifier adapter checkpoint directory.",
    )
    parser.add_argument(
        "--ref-checkpoint", type=str, default=None,
        help="Refiner adapter checkpoint directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-phase and per-example progress prints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    verbose = not args.quiet
    examples = load_Somadhan_split(args.devset, id_start=1)
    gt_answers = [ex.answer_target for ex in examples]

    if verbose:
        print(f"Loaded {len(examples)} examples from {args.devset}\n", flush=True)

    # --- Baseline 1: GanitLLM zero-shot (T→0, single pass) ---
    base_cfg = InferenceConfig(
        num_samples=1,
        temperature=0.01,
        show_progress=verbose,
    )
    mv_cfg = InferenceConfig(
        num_samples=args.num_samples,
        temperature=0.3,
        show_progress=verbose,
    )

    llama_cfg = MaltModelConfig()
    run_trained = bool(
        args.gen_checkpoint or args.ver_checkpoint or args.ref_checkpoint
    )
    total_phases = 4 if run_trained else 2

    if verbose:
        print("Loading base GanitLLM + adapter shells ...", flush=True)
    model, tok = load_malt_llama_with_adapters(llama_cfg)
    set_active_role_adapter(model, ROLE_GENERATOR)

    if verbose:
        print(
            f"\nPhase 1/{total_phases}: GanitLLM base zero-shot (1 sample / example) ...",
            flush=True,
        )
    base_preds = run_single_agent_generator(model, tok, examples, base_cfg)
    base_stats = evaluate_somadhan_predictions(base_preds, gt_answers)

    # --- Baseline 2: GanitLLM MV@3 (untrained adapters) ---
    if verbose:
        print(
            f"\nPhase 2/{total_phases}: GanitLLM base MV@{args.num_samples} (T=0.3) ...",
            flush=True,
        )
    mv_preds = run_single_agent_generator(model, tok, examples, mv_cfg)
    mv_stats = evaluate_somadhan_predictions(mv_preds, gt_answers)

    # --- If trained checkpoints provided, load them ---
    if run_trained:
        del model
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print("\nLoading trained adapter checkpoints ...", flush=True)
        model, tok = load_malt_llama_with_trained_adapters(
            cfg=llama_cfg,
            generator_checkpoint=args.gen_checkpoint,
            verifier_checkpoint=args.ver_checkpoint,
            refiner_checkpoint=args.ref_checkpoint,
        )

        # --- Trained single-agent (generator only) MV@3 ---
        set_active_role_adapter(model, ROLE_GENERATOR)
        if verbose:
            print(
                f"\nPhase 3/{total_phases}: Trained generator-only MV@{args.num_samples} ...",
                flush=True,
            )
        trained_single_preds = run_single_agent_generator(model, tok, examples, mv_cfg)
        trained_single_stats = evaluate_somadhan_predictions(trained_single_preds, gt_answers)

        # --- Full MALT multi-agent G→V→R MV@3 ---
        # run_multi_agent_malt swaps adapters internally before each stage,
        # so we can pass the same model object for all three roles.
        if verbose:
            print(
                f"\nPhase 4/{total_phases}: MALT G→V→R MV@{args.num_samples} ...",
                flush=True,
            )
        malt_preds = run_multi_agent_malt(model, model, model, tok, examples, mv_cfg)
        malt_stats = evaluate_somadhan_predictions(malt_preds, gt_answers)
    else:
        trained_single_stats = None
        malt_stats = None

    print("\n" + "=" * 60)
    print("Somadhan Devset Evaluation Results")
    print("=" * 60)
    print(f"GanitLLM base zero-shot (T≈0):   {base_stats.correct}/{base_stats.total}  acc={base_stats.accuracy:.4f}")
    print(f"GanitLLM base MV@{args.num_samples} (T=0.3):     {mv_stats.correct}/{mv_stats.total}  acc={mv_stats.accuracy:.4f}")

    if trained_single_stats:
        print(f"Trained G-only MV@{args.num_samples}:            {trained_single_stats.correct}/{trained_single_stats.total}  acc={trained_single_stats.accuracy:.4f}")
    if malt_stats:
        print(f"MALT G→V→R MV@{args.num_samples}:              {malt_stats.correct}/{malt_stats.total}  acc={malt_stats.accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
