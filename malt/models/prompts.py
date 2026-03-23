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

def build_generator_prompt(question: str) -> str:
    """
    Prompt for the Generator (G).

    Matches GanitLLM's training prompt exactly (Appendix E of ganitLLM.pdf)
    so the model stays in-distribution.
    """
    return (
        "A conversation takes place between the user and the assistant. "
        "The user asks a question, and the assistant solves the problem. "
        "Please reason step by step in Bengali, and put your final answer "
        "in the <answer> </answer> tags.\n\n"
        f"Question: {question}"
    )


def build_verifier_prompt(question: str, generator_output: str) -> str:
    """
    Prompt for the Verifier (V).

    The verifier re-checks the generator's reasoning and answer, identifies
    any errors, and provides its own answer inside <answer></answer> tags.
    """
    return dedent(
        f"""\
You are an expert solution checker for Bengali math problems.

Your task is to carefully read the problem and the proposed solution,
then verify whether the reasoning and final answer are correct.

- If the solution is correct, briefly explain why and keep the same
  final answer.
- If the solution is incorrect, explain the error, recompute the
  correct solution, and provide the correct final answer.

Reason step by step in Bengali. Always put your final answer in
the <answer> </answer> tags.

Problem:
{question}

Proposed solution:
{generator_output}"""
    )


def build_refiner_prompt(
    question: str,
    generator_output: str,
    verifier_output: str,
) -> str:
    """
    Prompt for the Refinement model (R).

    The refiner sees both the initial solution and the verification
    critique, and must produce a corrected final solution with the answer
    inside <answer></answer> tags.
    """
    return dedent(
        f"""\
You are an expert problem solver that refines math solutions based on
feedback.

You are given:
- A Bengali math word problem.
- An initial solution.
- A verification / critique of that solution.

Your task:
- Use all of this information to produce a clear, corrected, and
  concise final solution.
- Fix any mistakes in the original solution.
- Make sure the final answer is explicitly stated.

Reason step by step in Bengali. Always put your final answer in
the <answer> </answer> tags.

Problem:
{question}

Initial solution:
{generator_output}

Verification / critique:
{verifier_output}"""
    )


__all__ = [
    "format_chat_prompt",
    "format_chat_conversation",
    "build_generator_prompt",
    "build_verifier_prompt",
    "build_refiner_prompt",
]
