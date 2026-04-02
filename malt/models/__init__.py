"""
Model utilities for MALT.

The key entrypoint here is `load_malt_llama_with_adapters`, which loads a
4-bit-quantized Llama-3.1-8B-Instruct base model and attaches three LoRA
adapters for the Generator (G), Verifier (V), and Refiner (R) roles.
"""

from .base_model import (
    MaltModelConfig,
    load_ganit_llm_base,
    load_malt_llama_with_adapters,
    load_malt_llama_with_trained_adapters,
    set_active_role_adapter,
    ROLE_GENERATOR,
    ROLE_VERIFIER,
    ROLE_REFINER,
)

__all__ = [
    "MaltModelConfig",
    "load_ganit_llm_base",
    "load_malt_llama_with_adapters",
    "load_malt_llama_with_trained_adapters",
    "set_active_role_adapter",
    "ROLE_GENERATOR",
    "ROLE_VERIFIER",
    "ROLE_REFINER",
]

