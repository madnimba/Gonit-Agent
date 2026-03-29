from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import logging
import time

from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import TrainingArguments, PreTrainedTokenizerBase
from trl import DPOTrainer, DPOConfig
from malt.models import load_malt_llama_with_trained_adapters

from malt.data.preference_builders import (
    VerifierDpoSample,
    RefinerDpoSample,
    build_verifier_dpo_samples,
    build_refiner_dpo_samples,
)
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_adapters,
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


class DpoTextPairDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, str, str]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        prompt, chosen, rejected = self.samples[idx]
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }


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
        samples = samples[: max_train_samples]

    triples: List[Tuple[str, str, str]] = []
    times: List[float] = []
    total = len(samples)
    log.info("Building %d verifier DPO text triples", total)

    for idx, s in enumerate(samples):
        step_start = time.time()
        prompt = build_verifier_prompt(s.question, s.generator_output)
        triples.append((prompt.rstrip() + "\n\n", s.chosen.lstrip(), s.rejected.lstrip()))
        elapsed = time.time() - step_start
        times.append(elapsed)

        processed = idx + 1
        remaining = total - processed
        if times:
            avg = sum(times) / len(times)
            eta_s = avg * remaining
            h, r = divmod(int(eta_s), 3600)
            m, s = divmod(r, 60)
            eta_str = (
                f"{h}h {m}m {s}s" if h else
                f"{m}m {s}s" if m else
                f"{s}s"
            )
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
        samples = samples[: max_train_samples]

    triples: List[Tuple[str, str, str]] = []
    times: List[float] = []
    total = len(samples)
    log.info("Building %d refiner DPO text triples", total)

    for idx, s in enumerate(samples):
        step_start = time.time()
        prompt = build_refiner_prompt(
            s.question,
            s.generator_output,
            s.verifier_output,
        )
        triples.append((prompt + "\n\n", s.chosen.lstrip(), s.rejected.lstrip()))
        elapsed = time.time() - step_start
        times.append(elapsed)

        processed = idx + 1
        remaining = total - processed
        if times:
            avg = sum(times) / len(times)
            eta_s = avg * remaining
            h, r = divmod(int(eta_s), 3600)
            m, s = divmod(r, 60)
            eta_str = (
                f"{h}h {m}m {s}s" if h else
                f"{m}m {s}s" if m else
                f"{s}s"
            )
            log.info(
                "Refiner DPO Progress: %d/%d (%.1f%%) | Last: %.1fs | Avg: %.1fs | ETA: %s",
                processed, total, 100.0 * processed / total,
                elapsed, avg, eta_str,
            )

    return triples


def _build_dpo_trainer(
    model,
    ref_model,
    train_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DpoTrainingConfig,
):
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

def train_verifier_dpo(
    valued_trajectories_path: Path,
    cfg: DpoTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
    task: TaskName = "somadhan",
):
    """
    Train the Verifier adapter with DPO on valued trajectories.

    Loads the SFT-trained verifier checkpoint as both the policy model
    and the frozen reference model.
    """
    model_cfg = model_cfg or MaltModelConfig()

    model, tokenizer = load_malt_llama_with_trained_adapters(
        model_cfg,
        verifier_checkpoint=Path("checkpoints/verifier_sft"),
    )
    set_active_role_adapter(model, ROLE_VERIFIER)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ref_model, _ = load_malt_llama_with_trained_adapters(
        model_cfg,
        verifier_checkpoint=Path("checkpoints/verifier_sft"),
    )
    set_active_role_adapter(ref_model, ROLE_VERIFIER)
    ref_model.config.use_cache = False

    triples = _build_verifier_dpo_text_triples_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
        task=task,
    )
    dataset = HFDataset.from_dict(
        {
            "prompt": [format_chat_prompt(tokenizer, p) for p, _, _ in triples],
            "chosen": [c for _, c, _ in triples],
            "rejected": [r for _, _, r in triples],
        }
    )

    dpo_trainer = _build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
    )
    dpo_trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))

def train_refiner_dpo(
    valued_trajectories_path: Path,
    cfg: DpoTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
    task: TaskName = "somadhan",
):
    """
    Train the Refiner adapter with DPO on valued trajectories.

    Loads the SFT-trained *refiner* checkpoint as both the policy model
    and the frozen reference model.
    """
    model_cfg = model_cfg or MaltModelConfig()

    model, tokenizer = load_malt_llama_with_trained_adapters(
        model_cfg,
        refiner_checkpoint=Path("checkpoints/refiner_sft"),
    )
    set_active_role_adapter(model, ROLE_REFINER)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ref_model, _ = load_malt_llama_with_trained_adapters(
        model_cfg,
        refiner_checkpoint=Path("checkpoints/refiner_sft"),
    )
    set_active_role_adapter(ref_model, ROLE_REFINER)
    ref_model.config.use_cache = False

    triples = _build_refiner_dpo_text_triples_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
        task=task,
    )
    dataset = HFDataset.from_dict(
        {
            "prompt": [format_chat_prompt(tokenizer, p) for p, _, _ in triples],
            "chosen": [c for _, c, _ in triples],
            "rejected": [r for _, _, r in triples],
        }
    )

    dpo_trainer = _build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
    )
    dpo_trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))
