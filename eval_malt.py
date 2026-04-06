"""
Evaluate MALT-enhanced GanitLLM vs baseline on a Somadhan-style devset.

Phases (no majority vote for trained agents):
  1. GanitLLM zero-shot (HF weights, no LoRA), 1 sample/example (T≈0)
  2. GanitLLM majority-vote (HF weights, no LoRA), MV@k samples (T=0.3)
  3. MALT ablation: HF-base generator + trained verifier + refiner (1× chain, T≈0)
  4. MALT ablation: trained generator + refiner + HF-base verifier (1× chain, T≈0)
  5. MALT ablation: trained generator/verifier + HF-base refiner (1× chain, T≈0)
  6. MALT full: trained generator + verifier + refiner (1× chain, T≈0)

Phases 3–6 require all three checkpoints: --gen-checkpoint, --ver-checkpoint, --ref-checkpoint.

Resume + chunking:
  Use --resume to skip examples already present in per-phase caches under:
    <output-dir>/cache/<phase_tag>.jsonl
  Add --chunk-size N to evaluate missing examples in batches (for speed).

Answer extraction contract
--------------------------
All predictions stored in caches and passed to evaluate_somadhan_predictions()
are *pre-extracted, normalized answer strings* (e.g. "8", "22.5"), NOT raw
full model generations.  This means:

  - pipeline.py/_majority_vote() must return the best_norm string (already
    extracted + normalized), NOT the raw generation.
  - somadhan.py/extract_Somadhan_answer() must search for <answer> tags
    BEFORE stripping <think> blocks (so answers inside <think> are found).

If you are loading a cache written by an older version of this script that
stored raw generations, delete the cache directory and re-run without --resume.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from malt.data import load_Somadhan_split
from malt.data.somadhan import extract_Somadhan_answer, normalize_Somadhan_answer
from malt.inference.pipeline import InferenceConfig, run_multi_agent_malt, run_single_agent_generator
from malt.models import (
    MaltModelConfig,
    load_ganit_llm_base,
    load_malt_llama_with_trained_adapters,
)
from malt.utils.eval import EvalStats, evaluate_somadhan_predictions


CACHE_DIRNAME = "cache"
CACHE_SUFFIX = ".jsonl"


# ---------------------------------------------------------------------------
# Answer extraction helper
# ---------------------------------------------------------------------------

def _clean_pred(raw: str) -> str:
    """
    Convert a raw model generation (or already-extracted string) into a
    normalized answer string suitable for caching and evaluation.

    This is the single source of truth for answer extraction in this file.
    Calling this on an already-clean string (e.g. "8") is idempotent because
    normalize_Somadhan_answer("8") == "8".
    """
    return normalize_Somadhan_answer(extract_Somadhan_answer(raw))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MALT on Somadhan devset.")
    parser.add_argument("--devset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=3, help="MV@k samples for phase 2.")
    parser.add_argument("--gen-checkpoint", type=str, default=None, help="Generator adapter checkpoint directory.")
    parser.add_argument("--ver-checkpoint", type=str, default=None, help="Verifier adapter checkpoint directory.")
    parser.add_argument("--ref-checkpoint", type=str, default=None, help="Refiner adapter checkpoint directory.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Evaluate missing examples in batches of N.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip examples already present in per-phase cache files under --output-dir/cache/. "
            "NOTE: caches must have been written by this version of eval_malt.py (storing "
            "pre-extracted answers). Delete the cache dir if upgrading from an older version."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars and per-phase ETA logging.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase tag / cache helpers
# ---------------------------------------------------------------------------

def _phase_tag(phase_num: int) -> str:
    return {
        1: "phase_1_base_zero_shot",
        2: "phase_2_base_mv",
        3: "phase_3_ut_gen_trained_ver_ref",
        4: "phase_4_trained_gen_ref_ut_ver",
        5: "phase_5_trained_gen_ver_ut_ref",
        6: "phase_6_trained_gen_ver_ref",
    }[phase_num]


def _cache_path(out_dir: Path, phase_num: int) -> Path:
    return out_dir / CACHE_DIRNAME / f"{_phase_tag(phase_num)}{CACHE_SUFFIX}"


def _load_cache(cache_path: Path, n_expected: int) -> list[str | None]:
    """
    Load pre-extracted answer strings from a phase cache.

    Each line is: {"idx": <int>, "pred": "<extracted-answer-string>"}
    Values are already normalized answer strings, NOT raw generations.
    """
    preds: list[str | None] = [None] * n_expected
    if not cache_path.exists():
        return preds
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            idx  = rec.get("idx", None)
            pred = rec.get("pred", None)
            if isinstance(idx, int) and 0 <= idx < n_expected and isinstance(pred, str):
                # Re-normalize on load so caches written with slightly different
                # normalization are still comparable. Idempotent on clean strings.
                preds[idx] = _clean_pred(pred)
    return preds


def _append_cache_line(cache_path: Path, idx: int, pred: str) -> None:
    """
    Append one extracted-answer record to a phase cache.

    *pred* must already be a clean extracted+normalized answer string.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "pred": pred}, ensure_ascii=False) + "\n")
        f.flush()


def _chunked(indices: list[int], chunk_size: int) -> Iterable[list[int]]:
    for i in range(0, len(indices), chunk_size):
        yield indices[i : i + chunk_size]


def _log_phase_eta(verbose: bool, label: str, t_start: float, phase_times: list[float], total_phases: int) -> None:
    if not verbose:
        return
    elapsed = time.perf_counter() - t_start
    phase_times.append(elapsed)
    rem = total_phases - len(phase_times)
    msg = f"  {label} done in {elapsed:.1f}s"
    if rem > 0 and phase_times:
        avg = sum(phase_times) / len(phase_times)
        msg += f" | ETA ~{avg * rem:.0f}s for {rem} remaining phase(s)"
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def _eval_summary_lines(
    num_samples: int,
    base_stats: EvalStats,
    mv_stats: EvalStats,
    malt_ablation_stats: tuple[EvalStats, EvalStats, EvalStats, EvalStats] | None,
) -> list[str]:
    lines = [
        "=" * 60,
        "Somadhan Devset Evaluation Results",
        "=" * 60,
        (
            f"GanitLLM zero-shot HF, no LoRA (T≈0):   {base_stats.correct}/{base_stats.total}  "
            f"acc={base_stats.accuracy:.4f}"
        ),
        (
            f"GanitLLM majority-vote HF, no LoRA MV@{num_samples} (T=0.3):     "
            f"{mv_stats.correct}/{mv_stats.total}  acc={mv_stats.accuracy:.4f}"
        ),
    ]
    if malt_ablation_stats:
        s_ut_g, s_ut_v, s_ut_r, s_full = malt_ablation_stats
        lines.extend(
            [
                (
                    f"MALT HF-base gen + trained ver/ref (1×):   {s_ut_g.correct}/{s_ut_g.total}  "
                    f"acc={s_ut_g.accuracy:.4f}"
                ),
                (
                    f"MALT trained gen/ref + HF-base ver (1×):   {s_ut_v.correct}/{s_ut_v.total}  "
                    f"acc={s_ut_v.accuracy:.4f}"
                ),
                (
                    f"MALT trained gen/ver + HF-base ref (1×):   {s_ut_r.correct}/{s_ut_r.total}  "
                    f"acc={s_ut_r.accuracy:.4f}"
                ),
                (
                    f"MALT trained gen + ver + ref (1×):            {s_full.correct}/{s_full.total}  "
                    f"acc={s_full.accuracy:.4f}"
                ),
            ]
        )
    lines.append("=" * 60)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    examples = load_Somadhan_split(args.devset, id_start=1)
    # gt_answers are pre-extracted at load time by load_Somadhan_split →
    # extract_Somadhan_answer(answer_raw).  We normalize them here so that
    # comparison with our normalized preds is apples-to-apples.
    gt_answers = [normalize_Somadhan_answer(ex.answer_target) for ex in examples]
    n = len(examples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ckpt = bool(args.gen_checkpoint and args.ver_checkpoint and args.ref_checkpoint)
    any_ckpt = bool(args.gen_checkpoint or args.ver_checkpoint or args.ref_checkpoint)
    total_phases = 6 if all_ckpt else 2

    if any_ckpt and not all_ckpt and verbose:
        print(
            "Note: phases 3–6 require ALL three checkpoints. Running only phases 1–2 for this run.",
            flush=True,
        )

    base_preds: list[str | None] = [None] * n
    mv_preds:   list[str | None] = [None] * n
    s3_preds:   list[str | None] = [None] * n
    # s4_preds:   list[str | None] = [None] * n
    # s5_preds:   list[str | None] = [None] * n
    s6_preds:   list[str | None] = [None] * n

    if args.resume:
        base_preds = _load_cache(_cache_path(out_dir, 1), n)
        mv_preds   = _load_cache(_cache_path(out_dir, 2), n)
        if all_ckpt:
            s3_preds = _load_cache(_cache_path(out_dir, 3), n)
            # s4_preds = _load_cache(_cache_path(out_dir, 4), n)
            # s5_preds = _load_cache(_cache_path(out_dir, 5), n)
            s6_preds = _load_cache(_cache_path(out_dir, 6), n)

    if verbose:
        print(f"Loaded {n} examples from {args.devset}", flush=True)
        print(f"Output dir: {out_dir}\n", flush=True)

    phase_times: list[float] = []

    # ---------------- Phase 1 + Phase 2 (base model) ----------------
    need_phase1 = any(p is None for p in base_preds)
    need_phase2 = any(p is None for p in mv_preds)

    model = None
    tok   = None
    if need_phase1 or need_phase2:
        model, tok = load_ganit_llm_base(MaltModelConfig())

    if verbose:
        print(f"Phase 1/{total_phases}: zero-shot", flush=True)
    if need_phase1:
        cfg1    = InferenceConfig(num_samples=1, temperature=0.01, show_progress=False)
        missing = [i for i, p in enumerate(base_preds) if p is None]
        t_phase = time.perf_counter()
        for idx_chunk in tqdm(list(_chunked(missing, args.chunk_size)), disable=not verbose, desc="phase 1", unit="chunk"):
            subset      = [examples[i] for i in idx_chunk]
            # pipeline returns already-extracted normalized strings (post Bug 2 fix)
            preds_chunk = run_single_agent_generator(model, tok, subset, cfg1)
            for local_pos, i in enumerate(idx_chunk):
                # _clean_pred is idempotent: safe whether pipeline returns raw or clean
                pred = _clean_pred(preds_chunk[local_pos])
                base_preds[i] = pred
                _append_cache_line(_cache_path(out_dir, 1), i, pred)
        _log_phase_eta(verbose, "Phase 1", t_phase, phase_times, total_phases)
    elif verbose:
        print("Skipping phase 1 (all cached).", flush=True)

    if verbose:
        print(f"Phase 2/{total_phases}: majority-vote MV@{args.num_samples}", flush=True)
    if need_phase2:
        cfg2    = InferenceConfig(num_samples=args.num_samples, temperature=0.3, show_progress=False)
        missing = [i for i, p in enumerate(mv_preds) if p is None]
        t_phase = time.perf_counter()
        for idx_chunk in tqdm(list(_chunked(missing, args.chunk_size)), disable=not verbose, desc="phase 2", unit="chunk"):
            subset      = [examples[i] for i in idx_chunk]
            preds_chunk = run_single_agent_generator(model, tok, subset, cfg2)
            for local_pos, i in enumerate(idx_chunk):
                pred = _clean_pred(preds_chunk[local_pos])
                mv_preds[i] = pred
                _append_cache_line(_cache_path(out_dir, 2), i, pred)
        _log_phase_eta(verbose, "Phase 2", t_phase, phase_times, total_phases)
    elif verbose:
        print("Skipping phase 2 (all cached).", flush=True)

    assert all(p is not None for p in base_preds), "base_preds has None entries after phase 1"
    assert all(p is not None for p in mv_preds),   "mv_preds has None entries after phase 2"

    # Predictions are already clean normalized strings; gt_answers are also
    # normalized above.  evaluate_somadhan_predictions → Somadhan_exact_match
    # runs extract+normalize again, which is idempotent on clean strings.
    base_stats = evaluate_somadhan_predictions(list(base_preds), gt_answers)
    mv_stats   = evaluate_somadhan_predictions(list(mv_preds),   gt_answers)

    malt_ablation_stats: tuple[EvalStats, EvalStats, EvalStats, EvalStats] | None = None

    # ---------------- Phases 3–6 (MALT ablations) ----------------
    if all_ckpt:
        import gc
        import torch

        if model is not None:
            del model, tok
            model, tok = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def run_malt_phase(
            phase_num: int,
            desc: str,
            generator_ckpt: str | None,
            verifier_ckpt: str | None,
            refiner_ckpt: str | None,
            pred_store: list[str | None],
        ) -> EvalStats:
            missing = [i for i, p in enumerate(pred_store) if p is None]
            if not missing:
                if verbose:
                    print(f"Skipping phase {phase_num} ({desc}) (all cached).", flush=True)
                return evaluate_somadhan_predictions(list(pred_store), gt_answers)

            if verbose:
                print(f"Phase {phase_num}/{total_phases}: {desc} (missing {len(missing)})", flush=True)

            cfg     = InferenceConfig(num_samples=1, temperature=0.01, show_progress=False)
            model2, tok2 = load_malt_llama_with_trained_adapters(
                cfg=MaltModelConfig(),
                generator_checkpoint=generator_ckpt,
                verifier_checkpoint=verifier_ckpt,
                refiner_checkpoint=refiner_ckpt,
            )
            cache_path = _cache_path(out_dir, phase_num)
            t_phase    = time.perf_counter()
            for idx_chunk in tqdm(list(_chunked(missing, args.chunk_size)), disable=not verbose, desc=f"phase {phase_num}", unit="chunk"):
                subset      = [examples[i] for i in idx_chunk]
                preds_chunk = run_multi_agent_malt(
                    model2, model2, model2, tok2, subset, cfg,
                    use_lora_generator=generator_ckpt is not None,
                    use_lora_verifier=verifier_ckpt  is not None,
                    use_lora_refiner=refiner_ckpt    is not None,
                )
                for local_pos, i in enumerate(idx_chunk):
                    pred = _clean_pred(preds_chunk[local_pos])
                    pred_store[i] = pred
                    _append_cache_line(cache_path, i, pred)
            _log_phase_eta(verbose, f"Phase {phase_num}", t_phase, phase_times, total_phases)

            stats = evaluate_somadhan_predictions(list(pred_store), gt_answers)
            del model2, tok2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return stats

        s3 = run_malt_phase(3, "HF-base gen + trained ver/ref",  None,                args.ver_checkpoint, args.ref_checkpoint, s3_preds)
        # s4 = run_malt_phase(4, "trained gen/ref + HF-base ver",  args.gen_checkpoint, None,                args.ref_checkpoint, s4_preds)
        # s5 = run_malt_phase(5, "trained gen/ver + HF-base ref",  args.gen_checkpoint, args.ver_checkpoint, None,                s5_preds)
        s6 = run_malt_phase(6, "trained gen + ver + ref",        args.gen_checkpoint, args.ver_checkpoint, args.ref_checkpoint, s6_preds)

        # malt_ablation_stats = (s3, s4, s5, s6)
        malt_ablation_stats = (s3, s6)

    summary      = _eval_summary_lines(args.num_samples, base_stats, mv_stats, malt_ablation_stats)
    results_path = out_dir / "eval_results.txt"
    header       = [f"devset: {args.devset}", f"num_samples: {args.num_samples}", f"chunk_size: {args.chunk_size}", ""]
    results_path.write_text("\n".join(header + summary) + "\n", encoding="utf-8")

    if verbose:
        print("\n" + "\n".join(summary), flush=True)
        print(f"Wrote evaluation summary to {results_path}", flush=True)


if __name__ == "__main__":
    main()