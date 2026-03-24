"""
Role-specific prompt templates for the MALT pipeline.

These prompts are designed primarily for math-style reasoning tasks such as
GSM8K and MATH, but are written generically so they can be adapted later.
"""

from __future__ import annotations

from textwrap import dedent


def build_generator_prompt(question: str) -> str:
    """
    Prompt for the Generator (G).

    The generator should produce a detailed chain-of-thought solution and end
    with a clearly marked final answer line of the form:

        Final Answer: <answer>
    """
    return dedent(
        f"""
        You are an expert math problem solver.
        Solve the following Bengali math problem step by step, explaining your reasoning
        clearly. At the end, output a single line of the form. :

        Final Answer: <answer>
        Respond in Bengali

        Problem:
        {question}
        """
    ).strip()


def build_verifier_prompt(question: str, generator_output: str) -> str:
    """
    Prompt for the Verifier (V).

    The verifier should re-check the generator's reasoning and answer. It
    should either confirm correctness or identify mistakes and compute the
    correct answer, again ending with:

        Verdict: correct/incorrect
        Final Answer: <answer>
    Respond in Bengali.
    """
    return dedent(
        f"""
        You are an expert solution checker.

        Your task is to carefully read the problem and the proposed solution,
        then verify whether the reasoning and final answer are correct.

        - If the solution is correct, briefly explain why and keep the same
          final answer.
        - If the solution is incorrect, explain the error, recompute the
          correct solution, and provide the correct final answer.

        Always end your response with:
        Verdict: correct/incorrect
        Final Answer: <answer>
        Respond in Bengali.

        Problem:
        {question}

        Proposed solution:
        {generator_output}
        """
    ).strip()


def build_refiner_prompt(
    question: str,
    generator_output: str,
    verifier_output: str,
) -> str:
    """
    Prompt for the Refinement model (R).

    The refiner sees both the initial solution and the verification critique,
    and must produce a high-quality final solution with a clearly marked
    final answer line:

        Final Answer: <answer>
    """
    return dedent(
        f"""
        You are an expert problem solver that refines solutions based on
        feedback.

        You are given:
        - A math word problem.
        - An initial solution.
        - A verification/critique of that solution.

        Your task:
        - Use all of this information to produce a clear, corrected, and
          concise final solution.
        - Fix any mistakes in the original solution.
        - Make sure the final answer is explicitly stated.

        Always end your response with:
        Final Answer: <answer>
        Respond in Bengali

        Problem:
        {question}

        Initial solution:
        {generator_output}

        Verification / critique:
        {verifier_output}
        """
    ).strip()


__all__ = [
    "build_generator_prompt",
    "build_verifier_prompt",
    "build_refiner_prompt",
]

