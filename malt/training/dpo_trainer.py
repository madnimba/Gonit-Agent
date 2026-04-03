from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import logging
import time

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from malt.data.preference_builders import (
    VerifierDpoSample,
    RefinerDpoSample,
    build_verifier_dpo_samples,
    build_refiner_dpo_samples,
)
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_trained_adapters,
    set_active_role_adapter,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.models.prompts import (
    format_chat_prompt,
    build_verifier_prompt,
    build_refiner_prompt,
)
from malt.search.value_iteration import (
    TaskName,
    ValueIterationConfig,
    apply_value_iteration_to_trajectories,
)
from malt.utils.io import read_jsonl

log = logging.getLogger(__name__)


@dataclass
class DpoTrainingConfig:
    output_dir: Path
    beta: float = 0.2

    # SFT checkpoint paths used as the frozen reference model for DPO.
    # Must be set explicitly so callers cannot accidentally train against
    # the wrong reference model.
    verifier_sft_checkpoint: Path = field(default_factory=lambda: Path("checkpoints/verifier_sft"))
    refiner_sft_checkpoint: Path = field(default_factory=lambda: Path("checkpoints/refiner_sft"))

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    max_seq_length: int = 2048
    max_train_samples: int | None = None

    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 2

    bf16: bool = True
    fp16: bool = False


def _format_eta(eta_s: float) -> str:
    """Format a duration in seconds into a human-readable ETA string."""
    h, remainder = divmod(int(eta_s), 3600)
    mins, secs = divmod(remainder, 60)
    if h:
        return f"{h}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _build_verifier_dpo_text_triples_from_trajectories(
    valued_trajectories_path: Path,
    max_train_samples: int | None,
    task: TaskName = "somadhan",
) -> List[Tuple[str, str, str]]:
    trajs = read_jsonl(valued_trajectories_path)
    valued = apply_value_iteration_to_trajectories(
        trajectories=trajs,
        cfg=ValueIterationConfig(task=task),
    )
    samples: List[VerifierDpoSample] = build_verifier_dpo_samples(valued)

    if max_train_samples is not None:
        samples = samples[:max_train_samples]

    triples: List[Tuple[str, str, str]] = []
    times: List[float] = []
    total = len(samples)
    log.info("Building %d verifier DPO text triples", total)

    for idx, sample in enumerate(samples):
        step_start = time.time()
        prompt = build_verifier_prompt(sample.question, sample.generator_output)
        triples.append((prompt.rstrip() + "\n\n", sample.chosen.lstrip(), sample.rejected.lstrip()))
        elapsed = time.time() - step_start
        times.append(elapsed)

        processed = idx + 1
        remaining = total - processed
        avg = sum(times) / len(times)
        eta_str = _format_eta(avg * remaining)
        log.info(
            "Verifier DPO Progress: %d/%d (%.1f%%) | Last: %.1fs | Avg: %.1fs | ETA: %s",
            processed, total, 100.0 * processed / total,
            elapsed, avg, eta_str,
        )

    return triples


def _build_refiner_dpo_text_triples_from_trajectories(
    valued_trajectories_path: Path,
    max_train_samples: int | None,
    task: TaskName = "somadhan",
) -> List[Tuple[str, str, str]]:
    trajs = read_jsonl(valued_trajectories_path)
    valued = apply_value_iteration_to_trajectories(
        trajectories=trajs,
        cfg=ValueIterationConfig(task=task),
    )
    samples: List[RefinerDpoSample] = build_refiner_dpo_samples(valued)

    if max_train_samples is not None:
        samples = samples[:max_train_samples]

    triples: List[Tuple[str, str, str]] = []
    times: List[float] = []
    total = len(samples)
    log.info("Building %d refiner DPO text triples", total)

    for idx, sample in enumerate(samples):
        step_start = time.time()
        prompt = build_refiner_prompt(
            sample.question,
            sample.generator_output,
            sample.verifier_output,
        )
        triples.append((prompt + "\n\n", sample.chosen.lstrip(), sample.rejected.lstrip()))
        elapsed = time.time() - step_start
        times.append(elapsed)

        processed = idx + 1
        remaining = total - processed
        avg = sum(times) / len(times)
        eta_str = _format_eta(avg * remaining)
        log.info(
            "Refiner DPO Progress: %d/%d (%.1f%%) | Last: %.1fs | Avg: %.1fs | ETA: %s",
            processed, total, 100.0 * processed / total,
            elapsed, avg, eta_str,
        )

    return triples


def _build_dpo_trainer(
    model,
    ref_model,
    train_dataset: HFDataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DpoTrainingConfig,
) -> DPOTrainer:
    dpo_args = DPOConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        report_to=[],
        max_length=cfg.max_seq_length,
        beta=cfg.beta,
        gradient_checkpointing=True,
        use_cache=False,
    )

    return DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )


def _run_dpo(
    model,
    ref_model,
    tokenizer: PreTrainedTokenizerBase,
    triples: List[Tuple[str, str, str]],
    cfg: DpoTrainingConfig,
) -> None:
    """Format triples into an HFDataset, train, and save model + tokenizer."""
    dataset = HFDataset.from_dict(
        {
            "prompt": [format_chat_prompt(tokenizer, p) for p, _, _ in triples],
            "chosen": [c for _, c, _ in triples],
            "rejected": [r for _, _, r in triples],
        }
    )

    # Ensure output directory exists before training begins.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dpo_trainer = _build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
    )
    dpo_trainer.train()

    model.save_pretrained(str(cfg.output_dir))
    # Save tokenizer alongside the model so the checkpoint is self-contained
    # for inference without requiring the caller to supply it separately.
    tokenizer.save_pretrained(str(cfg.output_dir))


def train_verifier_dpo(
    valued_trajectories_path: Path,
    cfg: DpoTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
    task: TaskName = "somadhan",
) -> None:
    """
    Train the Verifier adapter with DPO on valued trajectories.

    Loads the SFT-trained verifier checkpoint (cfg.verifier_sft_checkpoint)
    as both the policy model and the frozen reference model.
    """
    model_cfg = model_cfg or MaltModelConfig()

    model, tokenizer = load_malt_llama_with_trained_adapters(
        model_cfg,
        verifier_checkpoint=cfg.verifier_sft_checkpoint,
    )
    set_active_role_adapter(model, ROLE_VERIFIER)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ref_model, _ = load_malt_llama_with_trained_adapters(
        model_cfg,
        verifier_checkpoint=cfg.verifier_sft_checkpoint,
    )
    set_active_role_adapter(ref_model, ROLE_VERIFIER)
    ref_model.config.use_cache = False

    triples = _build_verifier_dpo_text_triples_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
        task=task,
    )
    _run_dpo(model, ref_model, tokenizer, triples, cfg)


def train_refiner_dpo(
    valued_trajectories_path: Path,
    cfg: DpoTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
    task: TaskName = "somadhan",
) -> None:
    """
    Train the Refiner adapter with DPO on valued trajectories.

    Loads the SFT-trained refiner checkpoint (cfg.refiner_sft_checkpoint)
    as both the policy model and the frozen reference model.
    """
    model_cfg = model_cfg or MaltModelConfig()

    model, tokenizer = load_malt_llama_with_trained_adapters(
        model_cfg,
        refiner_checkpoint=cfg.refiner_sft_checkpoint,
    )
    set_active_role_adapter(model, ROLE_REFINER)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ref_model, _ = load_malt_llama_with_trained_adapters(
        model_cfg,
        refiner_checkpoint=cfg.refiner_sft_checkpoint,
    )
    set_active_role_adapter(ref_model, ROLE_REFINER)
    ref_model.config.use_cache = False

    triples = _build_refiner_dpo_text_triples_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
        task=task,
    )
    _run_dpo(model, ref_model, tokenizer, triples, cfg)