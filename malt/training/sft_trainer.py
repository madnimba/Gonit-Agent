from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)

from malt.data.preference_builders import (
    GeneratorSftSample,
    VerifierSftSample,
    RefinerSftSample,
    build_generator_sft_samples,
    build_verifier_and_refiner_sft_samples,
)
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_adapters,
    set_active_role_adapter,
    ROLE_GENERATOR,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.models.prompts import (
    format_chat_conversation,
    build_generator_prompt,
    build_verifier_prompt,
    build_refiner_prompt,
)
from malt.search.value_iteration import (
    TaskName,
    ValueIterationConfig,
    apply_value_iteration_to_trajectories,
)
from malt.utils.io import read_jsonl


@dataclass
class SftTrainingConfig:
    output_dir: Path

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    max_seq_length: int = 2048
    max_train_samples: int | None = None

    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 2

    bf16: bool = True
    fp16: bool = False


class SupervisedTextDataset(Dataset):
    """
    Supervised dataset wrapping (prompt_text, full_text) pairs where both
    have already been formatted through the chat template.

    *prompt_text* is everything up to the assistant generation marker;
    *full_text* is the complete conversation including the assistant
    response.  We tokenize *full_text* and mask the *prompt_text* portion
    (by token count) with -100 in the labels so the loss is computed only
    on the response.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pairs: Sequence[Tuple[str, str]],
        max_seq_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.pairs = list(pairs)
        self.max_seq_length = max_seq_length

        self._prompt_token_lens: List[int] = []
        for prompt_text, _ in self.pairs:
            enc = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
            self._prompt_token_lens.append(len(enc["input_ids"]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        _, full_text = self.pairs[idx]
        prompt_len = self._prompt_token_lens[idx]

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _build_generator_text_pairs_from_trajectories(
    valued_trajectories_path: Path,
    max_train_samples: int | None,
    task: TaskName = "somadhan",
) -> List[Tuple[str, str]]:
    trajs = read_jsonl(valued_trajectories_path)

    valued = apply_value_iteration_to_trajectories(
        trajectories=trajs,
        cfg=ValueIterationConfig(task=task),
    )

    samples: List[GeneratorSftSample] = build_generator_sft_samples(valued)

    if max_train_samples is not None:
        samples = samples[: max_train_samples]

    text_pairs: List[Tuple[str, str]] = []
    for s in samples:
        prompt = build_generator_prompt(s.question)
        response = s.generator_output
        text_pairs.append((prompt, response))

    return text_pairs


def _build_verifier_and_refiner_text_pairs_from_trajectories(
    valued_trajectories_path: Path,
    max_train_samples: int | None,
    task: TaskName = "somadhan",
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    trajs = read_jsonl(valued_trajectories_path)

    valued = apply_value_iteration_to_trajectories(
        trajectories=trajs,
        cfg=ValueIterationConfig(task=task),
    )

    v_samples, r_samples = build_verifier_and_refiner_sft_samples(valued)

    if max_train_samples is not None:
        v_samples = v_samples[: max_train_samples]
        r_samples = r_samples[: max_train_samples]

    v_pairs: List[Tuple[str, str]] = []
    for s in v_samples:
        prompt = build_verifier_prompt(s.question, s.generator_output)
        response = s.verifier_output
        v_pairs.append((prompt, response))

    r_pairs: List[Tuple[str, str]] = []
    for s in r_samples:
        prompt = build_refiner_prompt(
            s.question,
            s.generator_output,
            s.verifier_output,
        )
        response = s.refiner_output
        r_pairs.append((prompt, response))

    return v_pairs, r_pairs


def _run_sft(
    model,
    tokenizer,
    text_pairs: List[Tuple[str, str]],
    cfg: SftTrainingConfig,
) -> None:
    # Format (raw_prompt, response) → (prompt_text, full_text) via chat
    # template so training inputs match inference-time formatting.
    formatted_pairs: List[Tuple[str, str]] = []
    for raw_prompt, response in text_pairs:
        prompt_text, full_text = format_chat_conversation(
            tokenizer, raw_prompt, response,
        )
        formatted_pairs.append((prompt_text, full_text))

    dataset = SupervisedTextDataset(
        tokenizer=tokenizer,
        pairs=formatted_pairs,
        max_seq_length=cfg.max_seq_length,
    )

    training_args = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))


def train_generator_sft(
    valued_trajectories_path: Path,
    cfg: SftTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
    task: TaskName = "somadhan",
):
    model_cfg = model_cfg or MaltModelConfig()
    model, tokenizer = load_malt_llama_with_adapters(model_cfg)
    set_active_role_adapter(model, ROLE_GENERATOR)

    text_pairs = _build_generator_text_pairs_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
        task=task,
    )
    _run_sft(model, tokenizer, text_pairs, cfg)


def train_verifier_sft(
    valued_trajectories_path: Path,
    cfg: SftTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
    task: TaskName = "somadhan",
):
    model_cfg = model_cfg or MaltModelConfig()
    model, tokenizer = load_malt_llama_with_adapters(model_cfg)
    set_active_role_adapter(model, ROLE_VERIFIER)

    v_pairs, _ = _build_verifier_and_refiner_text_pairs_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
        task=task,
    )
    _run_sft(model, tokenizer, v_pairs, cfg)


def train_refiner_sft(
    valued_trajectories_path: Path,
    cfg: SftTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
    task: TaskName = "somadhan",
):
    model_cfg = model_cfg or MaltModelConfig()
    model, tokenizer = load_malt_llama_with_adapters(model_cfg)
    set_active_role_adapter(model, ROLE_REFINER)

    _, r_pairs = _build_verifier_and_refiner_text_pairs_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
        task=task,
    )
    _run_sft(model, tokenizer, r_pairs, cfg)
