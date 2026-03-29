"""
Evaluate MALT-enhanced GanitLLM vs baseline on Somadhan devset.

Reports four configurations:
  1. GanitLLM base zero-shot (single pass, T=0.0)
  2. GanitLLM base MV@3 (majority vote, T=0.3)
  3. Single-agent generator-only with trained adapter (MV@3)
  4. Multi-agent MALT  G→V→R with trained adapters (MV@3)

Usage (from repo root; ensure ``malt`` is importable, e.g. ``PYTHONPATH=.``):

  PYTHONPATH=. python scripts/eval_malt.py --devset data/somadhan_dev.csv

  # With trained checkpoints:
  PYTHONPATH=. python scripts/eval_malt.py --devset data/somadhan_dev.csv \\
      --gen-checkpoint checkpoints/generator_sft \\
      --ver-checkpoint checkpoints/verifier_dpo \\
      --ref-checkpoint checkpoints/refiner_dpo

  # Suppress progress lines:
  PYTHONPATH=. python scripts/eval_malt.py --devset data/dev.csv --quiet

  # Per-example JSONL (question + labeled phases with raw + extracted answer):
  PYTHONPATH=. python scripts/eval_malt.py --devset data/dev.csv \\
      --dump-predictions output/eval_predictions.jsonl \\
      --gen-checkpoint checkpoints/generator_sft \\
      --ver-checkpoint checkpoints/verifier_dpo \\
      --ref-checkpoint checkpoints/refiner_dpo
"""

import argparse
import json
from pathlib import Path

from malt.data import extract_Somadhan_answer, load_Somadhan_split
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
    parser.add_argument(
        "--dump-predictions",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write JSONL: each line is one example with `question`, `ground_truth`, "
            "and nested `phase_*` objects (raw model text + extracted answer)."
        ),
    )
    return parser.parse_args()


def _phase_record(title: str, raw_text: str) -> dict:
    return {
        "phase_title": title,
        "model_output_raw": raw_text,
        "answer_extracted": extract_Somadhan_answer(raw_text),
    }


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

    trained_single_preds: list[str] | None = None
    malt_preds: list[str] | None = None

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

    if args.dump_predictions:
        out = Path(args.dump_predictions)
        out.parent.mkdir(parents=True, exist_ok=True)
        p1 = (
            f"Phase 1/{total_phases}: GanitLLM base zero-shot "
            f"(T≈0, 1 sample / example)"
        )
        p2 = (
            f"Phase 2/{total_phases}: GanitLLM base MV@{args.num_samples} "
            f"(T=0.3, untrained LoRAs)"
        )
        p3 = (
            f"Phase 3/{total_phases}: Trained generator-only MV@{args.num_samples}"
        )
        p4 = (
            f"Phase 4/{total_phases}: MALT G→V→R MV@{args.num_samples}"
        )
        with out.open("w", encoding="utf-8") as f:
            for i, ex in enumerate(examples):
                record: dict = {
                    "id": ex.id,
                    "question": ex.question,
                    "ground_truth": gt_answers[i],
                    "phase_1_base_zero_shot": _phase_record(p1, base_preds[i]),
                    "phase_2_base_mv": _phase_record(p2, mv_preds[i]),
                }
                if trained_single_preds is not None:
                    record["phase_3_trained_generator_only"] = _phase_record(
                        p3, trained_single_preds[i]
                    )
                if malt_preds is not None:
                    record["phase_4_malt_g_v_r"] = _phase_record(
                        p4, malt_preds[i]
                    )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nWrote per-example predictions to {out}", flush=True)

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
