"""
Role-specific prompt templates for the MALT pipeline.

Prompts are designed for GanitLLM (Qwen3-4B based) on Bengali math
reasoning tasks.  The model expects answers inside <answer></answer> tags
and reasons step-by-step in Bengali.
"""

from __future__ import annotations

from textwrap import dedent
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Chat-template helper
# ---------------------------------------------------------------------------

def format_chat_prompt(tokenizer: PreTrainedTokenizerBase, user_content: str) -> str:
    """
    Wrap a user-message string in the model's chat template so that the
    tokenizer produces the correct special-token framing (e.g. Qwen3's
    ``<|im_start|>user ...``).

    Disables Qwen3's thinking mode so the model produces direct answers
    matching GanitLLM's training distribution (concise Bengali reasoning
    without ``<think>`` blocks).

    Returns a ready-to-tokenize string.
    """
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def format_chat_conversation(
    tokenizer: PreTrainedTokenizerBase,
    user_content: str,
    assistant_content: str,
) -> tuple[str, str]:
    """
    Format a complete user→assistant conversation through the chat template.

    Returns ``(prompt_text, full_text)`` where *prompt_text* is everything
    up to (and including) the assistant generation marker, and *full_text*
    is the complete conversation.  Used by SFT to compute the prompt-mask
    boundary so that training loss is only on the assistant response.
    """
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False,
        )
    return prompt_text, full_text


# ---------------------------------------------------------------------------
# Role prompts — return the raw *user-message content* (not chat-wrapped).
# Callers must pass through format_chat_prompt() before tokenization.
# ---------------------------------------------------------------------------

# The preamble is GanitLLM's exact training prompt (Appendix E of
# ganitLLM.pdf).  All three roles reuse it so the model stays
# in-distribution and always produces step-by-step Bengali reasoning.
_PREAMBLE = (
    "A conversation takes place between the user and the assistant. "
    "The user asks a question, and the assistant solves the problem. "
    "Please reason step by step in Bengali, and put your final answer "
    "in the <answer> </answer> tags."
)


def build_generator_prompt(question: str) -> str:
    """
    Prompt for the Generator (G).

    Uses GanitLLM's preamble with a Bengali instruction reinforcing
    that all steps must be shown and the answer must be in <answer> tags.
    """
    return (
        f"{_PREAMBLE}\n\n"
        f"Question: {question}\n\n"
        f"প্রতিটি ধাপ বিস্তারিতভাবে বাংলায় দেখান এবং চূড়ান্ত উত্তর অবশ্যই "
        f"<answer> </answer> ট্যাগের মধ্যে দিন।"
    )


def build_verifier_prompt(question: str, generator_output: str) -> str:
    """
    Prompt for the Verifier (V).

    Same GanitLLM preamble, but the "question" now includes the original
    problem and a proposed solution to verify.  The model is asked to
    re-solve independently and point out any errors.
    """
    return (
        f"{_PREAMBLE}\n\n"
        f"Question: নিচের সমস্যাটি এবং প্রস্তাবিত সমাধানটি মনোযোগ দিয়ে পড়ুন। "
        f"সমাধানের প্রতিটি ধাপ যাচাই করুন। যদি কোনো ভুল থাকে, সঠিক সমাধান দিন। "
        f"যদি সমাধান সঠিক হয়, কেন সঠিক তা ব্যাখ্যা করুন।\n\n"
        f"সমস্যা:\n{question}\n\n"
        f"প্রস্তাবিত সমাধান:\n{generator_output}"
    )


def build_refiner_prompt(
    question: str,
    generator_output: str,
    verifier_output: str,
) -> str:
    """
    Prompt for the Refinement model (R).

    Same GanitLLM preamble, but the "question" now includes the original
    problem, initial solution, and a verification critique.  The model
    must produce a corrected final solution.
    """
    return (
        f"{_PREAMBLE}\n\n"
        f"Question: নিচের সমস্যা, প্রাথমিক সমাধান এবং যাচাইকরণ মনোযোগ দিয়ে পড়ুন। "
        f"সব তথ্য ব্যবহার করে একটি সঠিক ও সংক্ষিপ্ত চূড়ান্ত সমাধান তৈরি করুন। "
        f"প্রতিটি ধাপ স্পষ্টভাবে দেখান।\n\n"
        f"সমস্যা:\n{question}\n\n"
        f"প্রাথমিক সমাধান:\n{generator_output}\n\n"
        f"যাচাইকরণ:\n{verifier_output}"
    )


__all__ = [
    "format_chat_prompt",
    "format_chat_conversation",
    "build_generator_prompt",
    "build_verifier_prompt",
    "build_refiner_prompt",
]
