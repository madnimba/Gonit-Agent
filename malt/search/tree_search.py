from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Union

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from malt.data import Gsm8kExample
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_adapters,
    set_active_role_adapter,
    ROLE_GENERATOR,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.models.prompts import (
    build_generator_prompt,
    build_verifier_prompt,
    build_refiner_prompt,
)
from malt.data import SomadhanExample, load_Somadhan_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# Union type accepted everywhere a question object is expected.
AnyExample = Union[Gsm8kExample, SomadhanExample]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TreeSearchConfig:
    """Configuration for G→V→R tree search."""

    branching_factor: int = 3

    max_new_tokens_generator: int = 512
    max_new_tokens_verifier: int = 512
    max_new_tokens_refiner: int = 512

    temperature_generator: float = 0.7
    temperature_verifier: float = 0.5
    temperature_refiner: float = 0.5

    top_p: float = 0.95
    top_k: int = 50

    # Maximum sequences to send to model.generate() in one call.
    # Lower this if you hit OOM; raise it (up to n**3) to maximise GPU util.
    max_batch_size: int = 32

    # Whether to torch.compile the model after loading.
    # NOTE: set False on shared GPUs — compile spawns Triton worker processes
    # that each hold ~3 GiB of VRAM independently of the parent process.
    use_torch_compile: bool = False

    # Output path for JSONL trajectories (each line is a single question's tree).
    output_path: Path = Path("data/malt_trajectories.jsonl")


# ---------------------------------------------------------------------------
# Data classes for the search tree
# ---------------------------------------------------------------------------

@dataclass
class RefinerNode:
    text: str


@dataclass
class VerifierNode:
    text: str
    refiners: List[RefinerNode] = field(default_factory=list)


@dataclass
class GeneratorNode:
    text: str
    verifiers: List[VerifierNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core batched inference helper
# ---------------------------------------------------------------------------

# Module-level variable: the last successful batch size. Persists across
# _sample_texts calls within a run so OOM probing only happens once — on the
# first question — rather than repeating the 32→16→8 retry sequence for all
# 8792 questions.
_current_batch_size: Optional[int] = None


def _sample_texts(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_batch_size: int = 32,
) -> List[str]:
    """
    Sample one completion per prompt, batching up to *max_batch_size* prompts
    per forward pass.

    Speedups vs. naive implementation:
    - torch.inference_mode(): disables autograd tracking entirely, lower
      overhead than no_grad for pure inference.
    - use_cache=True: explicit KV cache so each autoregressive step only
      processes the new token rather than the full prefix.
    - Persistent batch size: _current_batch_size remembers the last successful
      size across calls so OOM retries only happen once per run, not once per
      question.

    On CUDA OOM, halves the batch size and retries down to a minimum of 1.
    """
    global _current_batch_size
    model.eval()
    all_outputs: List[str] = []

    # Start from the last known-good size, capped at the caller's max.
    if _current_batch_size is None:
        _current_batch_size = max_batch_size
    current_batch_size = min(_current_batch_size, max_batch_size)

    with torch.inference_mode():
        batch_start = 0
        while batch_start < len(prompts):
            batch = list(prompts[batch_start : batch_start + current_batch_size])

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # Catch both torch OOM and lower-level CUDA errors such as
                # CUBLAS_STATUS_ALLOC_FAILED, which surfaces as RuntimeError
                # rather than OutOfMemoryError when cuBLAS cannot allocate its
                # internal workspace due to insufficient VRAM.
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                    isinstance(exc, RuntimeError)
                    and any(
                        kw in str(exc)
                        for kw in ("CUBLAS_STATUS_ALLOC_FAILED", "CUDA out of memory",
                                   "out of memory", "CUBLAS_STATUS")
                    )
                )
                if not is_oom:
                    raise
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise
                new_size = max(1, current_batch_size // 2)
                log.warning(
                    "OOM during generate (batch_size=%d, error=%s). "
                    "Retrying with batch_size=%d.",
                    current_batch_size, type(exc).__name__, new_size,
                )
                current_batch_size = new_size
                continue  # retry same batch_start with smaller batch

            # Remember the size that worked so future calls start here.
            _current_batch_size = current_batch_size

            input_len = inputs["input_ids"].shape[1]
            for ids in generated_ids:
                text = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
                all_outputs.append(text.strip())

            batch_start += current_batch_size

    return all_outputs


# ---------------------------------------------------------------------------
# Tree search — role stages are each executed in a single batched pass
# ---------------------------------------------------------------------------

def run_tree_search_for_questions(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: TreeSearchConfig,
) -> Iterable[tuple[dict, float]]:
    """
    Run G→V→R tree search for a list of questions.

    Accepts both Gsm8kExample and SomadhanExample. Yields (trajectory, elapsed)
    tuples so the caller can track per-question timing for ETA estimates.
    """
    n = cfg.branching_factor

    for ex in questions:
        step_start = time.time()

        # ------------------------------------------------------------------
        # Stage 1: Generator — n prompts, 1 adapter swap, 1 batched call
        # ------------------------------------------------------------------
        set_active_role_adapter(model, ROLE_GENERATOR)

        gen_prompts = [build_generator_prompt(ex.question) for _ in range(n)]
        gen_texts: List[str] = _sample_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=gen_prompts,
            max_new_tokens=cfg.max_new_tokens_generator,
            temperature=cfg.temperature_generator,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_batch_size=cfg.max_batch_size,
        )

        # ------------------------------------------------------------------
        # Stage 2: Verifier — n² prompts, 1 adapter swap, 1 batched call
        # ------------------------------------------------------------------
        set_active_role_adapter(model, ROLE_VERIFIER)

        v_prompts: List[str] = []
        for g_text in gen_texts:
            for _ in range(n):
                v_prompts.append(build_verifier_prompt(ex.question, g_text))

        v_texts_flat: List[str] = _sample_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=v_prompts,
            max_new_tokens=cfg.max_new_tokens_verifier,
            temperature=cfg.temperature_verifier,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_batch_size=cfg.max_batch_size,
        )
        v_texts: List[List[str]] = [
            v_texts_flat[i * n : (i + 1) * n] for i in range(n)
        ]

        # ------------------------------------------------------------------
        # Stage 3: Refiner — n³ prompts, 1 adapter swap, 1 batched call
        # ------------------------------------------------------------------
        set_active_role_adapter(model, ROLE_REFINER)

        r_prompts: List[str] = []
        for g_idx, g_text in enumerate(gen_texts):
            for v_text in v_texts[g_idx]:
                for _ in range(n):
                    r_prompts.append(
                        build_refiner_prompt(ex.question, g_text, v_text)
                    )

        r_texts_flat: List[str] = _sample_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=r_prompts,
            max_new_tokens=cfg.max_new_tokens_refiner,
            temperature=cfg.temperature_refiner,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_batch_size=cfg.max_batch_size,
        )

        # ------------------------------------------------------------------
        # Reassemble the tree from the flat output lists
        # ------------------------------------------------------------------
        generator_nodes: List[GeneratorNode] = []
        r_idx = 0

        for g_idx, g_text in enumerate(gen_texts):
            verifier_nodes: List[VerifierNode] = []

            for v_text in v_texts[g_idx]:
                refiner_texts = r_texts_flat[r_idx : r_idx + n]
                r_idx += n
                refiner_nodes = [RefinerNode(text=t) for t in refiner_texts]
                verifier_nodes.append(
                    VerifierNode(text=v_text, refiners=refiner_nodes)
                )

            generator_nodes.append(
                GeneratorNode(text=g_text, verifiers=verifier_nodes)
            )

        trajectory = {
            "id": ex.id,
            "question": ex.question,
            "answer_gt": ex.answer_target,
            "generator_nodes": [
                {
                    "text": g.text,
                    "verifier_nodes": [
                        {
                            "text": v.text,
                            "refiner_nodes": [{"text": r.text} for r in v.refiners],
                        }
                        for v in g.verifiers
                    ],
                }
                for g in generator_nodes
            ],
        }

        yield trajectory, time.time() - step_start


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _get_last_processed_id(path: Path) -> Optional[str]:
    """
    Return the id of the last successfully written trajectory, or None.
    Reads only the final non-empty line so it stays O(1) for large files.
    """
    if not path.exists():
        return None

    last_line: Optional[str] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line

    if last_line is None:
        return None

    try:
        obj = json.loads(last_line)
        return obj.get("id")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared internal runner
# ---------------------------------------------------------------------------

def _run_tree_search(
    examples: List[AnyExample],
    cfg: TreeSearchConfig,
    model_cfg: Optional[MaltModelConfig] = None,
) -> Path:
    """
    Shared implementation: apply resumption, load model, optionally compile,
    run tree search, and write JSONL output with ETA logging.
    """
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Output path: %s", cfg.output_path)

    last_id = _get_last_processed_id(cfg.output_path)
    print("Last processed id:", last_id)

    if last_id is not None:
        before = len(examples)
        examples = [ex for ex in examples if int(ex.id) > int(last_id)]
        log.info(
            "Resuming after id %s — skipping %d already-processed questions",
            last_id,
            before - len(examples),
        )

    model, tokenizer = load_malt_llama_with_adapters(model_cfg)
    log.info("Model loaded")

    if cfg.use_torch_compile:
        log.info("Compiling model with torch.compile(mode='reduce-overhead')…")
        model = torch.compile(model, mode="reduce-overhead")
        log.info("Compilation done")

    total = len(examples)
    times: List[float] = []
    mode = "a" if cfg.output_path.exists() else "w"

    with cfg.output_path.open(mode, encoding="utf-8") as f:
        for idx, (traj, elapsed) in enumerate(
            run_tree_search_for_questions(
                model=model,
                tokenizer=tokenizer,
                questions=examples,
                cfg=cfg,
            )
        ):
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            f.flush()

            times.append(elapsed)
            processed = idx + 1
            remaining = total - processed
            avg = sum(times) / len(times)

            if remaining > 0:
                eta_s = avg * remaining
                h, r = divmod(int(eta_s), 3600)
                m, s = divmod(r, 60)
                eta_str = (
                    f"{h}h {m}m {s}s" if h else
                    f"{m}m {s}s" if m else
                    f"{s}s"
                )
                log.info(
                    "Progress: %d/%d (%.1f%%) | Last: %.1fs | Avg: %.1fs | "
                    "Batch: %s | ETA: %s",
                    processed, total, 100.0 * processed / total,
                    elapsed, avg,
                    _current_batch_size,
                    eta_str,
                )
            else:
                log.info(
                    "Progress: %d/%d (100.0%%) | Last: %.1fs | Avg: %.1fs",
                    processed, total, elapsed, avg,
                )

    log.info("Done. Trajectories written to %s", cfg.output_path)
    return cfg.output_path


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_tree_search_for_gsm8k_split(
    split: Literal["train", "test", "validation"],
    cfg: TreeSearchConfig,
    model_cfg: Optional[MaltModelConfig] = None,
) -> Path:
    """
    Load a GSM8K split and run tree search over it, writing one JSON line per
    question to cfg.output_path.
    """
    from malt.data import load_gsm8k_split  # imported lazily to avoid cycles

    examples = load_gsm8k_split(split)
    log.info("Loaded %d examples from GSM8K '%s' split", len(examples), split)
    return _run_tree_search(examples, cfg, model_cfg)


def run_tree_search_for_somadhan(
    csv_path: str | Path,
    cfg: TreeSearchConfig,
    model_cfg: Optional[MaltModelConfig] = None,
) -> Path:
    """
    Load the full Somadhan dataset from a CSV file and run tree search over it,
    writing one JSON line per question to cfg.output_path.
    """
    examples = load_Somadhan_split(csv_path)
    log.info("Loaded %d examples from Somadhan CSV '%s'", len(examples), csv_path)
    return _run_tree_search(examples, cfg, model_cfg)