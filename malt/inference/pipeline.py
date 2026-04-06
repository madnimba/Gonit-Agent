from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Union

import torch
from tqdm.auto import tqdm
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

    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

    num_samples: int = 3

    # If True, print one line per dev example during inference (off by default).
    show_progress: bool = False

    # Label for tqdm when show_progress is True (per-example progress + ETA).
    progress_desc: str | None = None


# ---------------------------------------------------------------------------
# Low-level generation helpers
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


def _generate_batch(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    """
    Generate one completion per prompt (batched).

    *prompts* should be raw user-message content strings (not chat-formatted).
    This function applies the chat template per prompt, batches tokenization,
    and decodes the generated continuation for each row.
    """
    if not prompts:
        return []

    formatted = [format_chat_prompt(tokenizer, p) for p in prompts]

    model.eval()
    with torch.inference_mode():
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
            use_cache=True,
        )

        # Per-row prompt lengths (works for left- or right-padding).
        attn = inputs.get("attention_mask", None)
        if attn is None:
            input_lens = [inputs["input_ids"].shape[1]] * gen_ids.shape[0]
        else:
            input_lens = attn.sum(dim=1).tolist()

        outputs: List[str] = []
        for row, in_len in zip(gen_ids, input_lens):
            text = tokenizer.decode(row[int(in_len):], skip_special_tokens=True)
            outputs.append(text.strip())
        return outputs


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
    # Return the extracted+normalized answer string directly,
    # NOT the raw full generation. Callers (eval) must not re-parse.
    return best_norm


def _majority_vote_and_raw(
    scored_raw_pairs: List[Tuple[str, str]],
    extract_fn: Callable[[str], str],
    normalize_fn: Callable[[str], str],
) -> Tuple[str, str]:
    """
    Majority vote using *scored_raw_pairs* (vote_text, raw_trace).

    *vote_text* is passed through extract+normalize for voting (same as
    ``_majority_vote``). *raw_trace* is the full model output to keep for logging
    (e.g. generator completion, or G→V→R sections for MALT).
    """
    if not scored_raw_pairs:
        return "", ""

    norm_counts: Counter = Counter(
        normalize_fn(extract_fn(vote_src)) for vote_src, _ in scored_raw_pairs
    )
    best_norm, _ = norm_counts.most_common(1)[0]
    for vote_src, raw_trace in scored_raw_pairs:
        if normalize_fn(extract_fn(vote_src)) == best_norm:
            return best_norm, raw_trace
    return best_norm, scored_raw_pairs[-1][1]


def _malt_gvr_trace(generator_text: str, verifier_text: str, refiner_text: str) -> str:
    return (
        f"### Generator\n{generator_text}\n\n"
        f"### Verifier\n{verifier_text}\n\n"
        f"### Refiner\n{refiner_text}"
    )


# ---------------------------------------------------------------------------
# Single-agent inference
# ---------------------------------------------------------------------------

def run_single_agent_generator(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: InferenceConfig,
    *,
    return_raw: bool = False,
) -> Union[List[str], List[Tuple[str, str]]]:
    final_answers: List[str] = []
    final_raws: List[str] = []
    n = len(questions)

    # Strategy: batch over examples, and if num_samples>1, do k batched passes.
    # This avoids a giant (batch_size * k) prompt list that can OOM.
    pbar = tqdm(
        total=n,
        desc=cfg.progress_desc or "Generator",
        unit="ex",
        disable=not cfg.show_progress,
    )

    # Process in chunks to keep GPU memory predictable.
    chunk_size = n  # default: one batch (callers can pass subsets for chunking)
    # Heuristic: if show_progress, tqdm is per-example anyway; chunking is still useful.
    # Callers can chunk at a higher layer by calling this function on subsets.
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = list(questions[start:end])

        # Collect k samples per example (answers_acc[i] is list of raw generations).
        answers_acc: List[List[str]] = [[] for _ in range(len(chunk))]
        for _ in range(cfg.num_samples):
            prompts = [build_generator_prompt(ex.question) for ex in chunk]
            gen_texts = _generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )
            for i, txt in enumerate(gen_texts):
                answers_acc[i].append(txt)

        for ex, answers in zip(chunk, answers_acc):
            extract_fn, normalize_fn = _get_answer_fns(ex)
            if return_raw:
                pairs = [(a, a) for a in answers]
                pred, raw = _majority_vote_and_raw(pairs, extract_fn, normalize_fn)
                final_answers.append(pred)
                final_raws.append(raw)
            else:
                final_answers.append(_majority_vote(answers, extract_fn, normalize_fn))

        pbar.update(len(chunk))
        start = end

    pbar.close()

    if return_raw:
        return list(zip(final_answers, final_raws, strict=True))
    return final_answers


run_single_agent_generator_gsm8k = run_single_agent_generator


# ---------------------------------------------------------------------------
# Multi-agent MALT inference
# ---------------------------------------------------------------------------

def _generate_malt_role(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    cfg: InferenceConfig,
    *,
    role: str,
    use_lora: bool,
) -> List[str]:
    """
    Run one MALT stage. If *use_lora* is False, run the frozen HF base (no LoRA),
    matching ``load_ganit_llm_base`` behavior instead of random init adapters.
    """
    if use_lora:
        set_active_role_adapter(model, role)
        return _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )
    if not hasattr(model, "disable_adapter"):
        raise TypeError(
            "Untrained-role MALT stages require PeftModel.disable_adapter(); upgrade peft."
        )
    with model.disable_adapter():
        return _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )


def run_multi_agent_malt(
    generator_model: PeftModel,
    verifier_model: PeftModel,
    refiner_model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: InferenceConfig,
    *,
    use_lora_generator: bool = True,
    use_lora_verifier: bool = True,
    use_lora_refiner: bool = True,
    return_raw: bool = False,
) -> Union[List[str], List[Tuple[str, str]]]:
    """
    MALT G→V→R chain. For each role, *use_lora_<role>* selects trained LoRA weights;
    if False, that stage runs on the HF base only (``disable_adapter()``), matching
    published GanitLLM without random adapter init.
    """
    final_answers: List[str] = []
    final_raws: List[str] = []
    n = len(questions)

    pbar = tqdm(
        total=n,
        desc=cfg.progress_desc or "MALT G→V→R",
        unit="ex",
        disable=not cfg.show_progress,
    )

    chunk_size = n  # callers can pass subsets for chunking
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = list(questions[start:end])

        # For each example, collect (refiner_output, trace). Trace is the full G→V→R
        # path when *return_raw*; otherwise trace equals the refiner string (no extra work).
        answers_acc: List[List[Tuple[str, str]]] = [[] for _ in range(len(chunk))]

        for _ in range(cfg.num_samples):
            # G stage
            g_prompts = [build_generator_prompt(ex.question) for ex in chunk]
            g_texts = _generate_malt_role(
                generator_model,
                tokenizer,
                g_prompts,
                cfg,
                role=ROLE_GENERATOR,
                use_lora=use_lora_generator,
            )

            # V stage
            v_prompts = [
                build_verifier_prompt(ex.question, g_text)
                for ex, g_text in zip(chunk, g_texts)
            ]
            v_texts = _generate_malt_role(
                verifier_model,
                tokenizer,
                v_prompts,
                cfg,
                role=ROLE_VERIFIER,
                use_lora=use_lora_verifier,
            )

            # R stage
            r_prompts = [
                build_refiner_prompt(ex.question, g_text, v_text)
                for ex, g_text, v_text in zip(chunk, g_texts, v_texts)
            ]
            r_texts = _generate_malt_role(
                refiner_model,
                tokenizer,
                r_prompts,
                cfg,
                role=ROLE_REFINER,
                use_lora=use_lora_refiner,
            )

            for i, (g_text, v_text, r_text) in enumerate(zip(g_texts, v_texts, r_texts)):
                trace = (
                    _malt_gvr_trace(g_text, v_text, r_text)
                    if return_raw
                    else r_text
                )
                answers_acc[i].append((r_text, trace))

        for ex, pairs in zip(chunk, answers_acc):
            extract_fn, normalize_fn = _get_answer_fns(ex)
            pred, raw_trace = _majority_vote_and_raw(pairs, extract_fn, normalize_fn)
            final_answers.append(pred)
            if return_raw:
                final_raws.append(raw_trace)

        pbar.update(len(chunk))
        start = end

    pbar.close()
    if return_raw:
        return list(zip(final_answers, final_raws, strict=True))
    return final_answers


run_multi_agent_malt_gsm8k = run_multi_agent_malt
