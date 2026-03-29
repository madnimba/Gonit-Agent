from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Sequence, Union

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from malt.data import Gsm8kExample, extract_gsm8k_answer, normalize_gsm8k_answer
from malt.models.prompts import (
    format_chat_prompt,
    build_generator_prompt,
    build_verifier_prompt,
    build_refiner_prompt,
)
from malt.models import (
    set_active_role_adapter,
    ROLE_GENERATOR,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.data import SomadhanExample, extract_Somadhan_answer, normalize_Somadhan_answer


AnyExample = Union[Gsm8kExample, SomadhanExample]


def _get_answer_fns(
    example: AnyExample,
) -> tuple[Callable[[str], str], Callable[[str], str]]:
    if isinstance(example, SomadhanExample):
        return extract_Somadhan_answer, normalize_Somadhan_answer
    return extract_gsm8k_answer, normalize_gsm8k_answer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Configuration for inference over any supported dataset."""

    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50

    num_samples: int = 3

    # If True, print one line per dev example during inference (off by default).
    show_progress: bool = False


# ---------------------------------------------------------------------------
# Low-level generation helper
# ---------------------------------------------------------------------------

def _generate_single(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    """
    Generate a single completion.

    *prompt* should be the raw user-message content — this function applies
    the chat template before tokenizing.
    """
    formatted = format_chat_prompt(tokenizer, prompt)

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
        )

        gen_text = tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return gen_text.strip()


# ---------------------------------------------------------------------------
# Majority-vote helper
# ---------------------------------------------------------------------------

def _majority_vote(
    answers: List[str],
    extract_fn: Callable[[str], str],
    normalize_fn: Callable[[str], str],
) -> str:
    if not answers:
        return ""

    norm_counts: Counter = Counter(
        normalize_fn(extract_fn(a)) for a in answers
    )
    best_norm, _ = norm_counts.most_common(1)[0]
    for a in answers:
        if normalize_fn(extract_fn(a)) == best_norm:
            return a
    return ""


# ---------------------------------------------------------------------------
# Single-agent inference
# ---------------------------------------------------------------------------

def run_single_agent_generator(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: InferenceConfig,
) -> List[str]:
    final_answers: List[str] = []

    for ex in questions:
        extract_fn, normalize_fn = _get_answer_fns(ex)
        answers: List[str] = []

        for i in range(cfg.num_samples):
            print(f"Solving {i}/{cfg.num_samples}")
            prompt = build_generator_prompt(ex.question)
            gen_text = _generate_single(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )
            answers.append(gen_text)

        final_answers.append(_majority_vote(answers, extract_fn, normalize_fn))

    return final_answers


run_single_agent_generator_gsm8k = run_single_agent_generator


# ---------------------------------------------------------------------------
# Multi-agent MALT inference
# ---------------------------------------------------------------------------

def run_multi_agent_malt(
    generator_model: PeftModel,
    verifier_model: PeftModel,
    refiner_model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: InferenceConfig,
) -> List[str]:
    final_answers: List[str] = []
    n = len(questions)

    for i, ex in enumerate(questions):
        if cfg.show_progress:
            print(
                f"  example {i + 1}/{n} (×{cfg.num_samples} G→V→R chains) ...",
                flush=True,
            )
        extract_fn, normalize_fn = _get_answer_fns(ex)
        answers: List[str] = []

        for _ in range(cfg.num_samples):
            set_active_role_adapter(generator_model, ROLE_GENERATOR)
            g_text = _generate_single(
                model=generator_model,
                tokenizer=tokenizer,
                prompt=build_generator_prompt(ex.question),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            set_active_role_adapter(verifier_model, ROLE_VERIFIER)
            v_text = _generate_single(
                model=verifier_model,
                tokenizer=tokenizer,
                prompt=build_verifier_prompt(ex.question, g_text),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            set_active_role_adapter(refiner_model, ROLE_REFINER)
            r_text = _generate_single(
                model=refiner_model,
                tokenizer=tokenizer,
                prompt=build_refiner_prompt(ex.question, g_text, v_text),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            answers.append(r_text)

        final_answers.append(_majority_vote(answers, extract_fn, normalize_fn))

    return final_answers


run_multi_agent_malt_gsm8k = run_multi_agent_malt
