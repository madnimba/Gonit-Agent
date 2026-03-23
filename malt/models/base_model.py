from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import logging
import os
import signal
import subprocess
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None  # type: ignore

from peft import LoraConfig, get_peft_model, PeftModel

log = logging.getLogger(__name__)

ROLE_GENERATOR: Literal["generator"] = "generator"
ROLE_VERIFIER: Literal["verifier"] = "verifier"
ROLE_REFINER: Literal["refiner"] = "refiner"

# GanitLLM-4B in 4-bit + three LoRA adapters needs roughly this much VRAM.
_MIN_FREE_VRAM_GIB = 4.0


@dataclass
class MaltModelConfig:
    """
    Configuration for loading the base model and attaching LoRA adapters.

    Default model is GanitLLM-4B (Qwen3-4B fine-tuned for Bengali math).
    4-bit quantization is optional — the 4B model fits in 24 GB at bf16,
    but 4-bit frees headroom for larger batch sizes during tree search.
    """

    model_name: str = "dipta007/GanitLLM-4B_SFT_CGRPO"
    load_in_4bit: bool = True
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16

    # LoRA configuration — rank 16 matches GanitLLM paper's GRPO LoRA.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def _gib(bytes_: int) -> float:
    return bytes_ / (1024 ** 3)


def _free_vram_gib(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, _ = torch.cuda.mem_get_info(device)
        return _gib(free)
    except Exception:
        return 0.0


def _get_other_gpu_pids(device: int = 0) -> list[int]:
    own_pid = os.getpid()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pids = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                pid = int(line)
                if pid != own_pid:
                    pids.append(pid)
        return pids
    except Exception:
        return []


def _kill_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            log.info("Killed stale GPU process %d", pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            log.warning("No permission to kill PID %d — skipping", pid)


def release_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log.info("GPU cache cleared. Free VRAM: %.1f GiB", _free_vram_gib())


def check_gpu_memory(
    min_free_gib: float = _MIN_FREE_VRAM_GIB,
    device: int = 0,
    auto_kill: bool = True,
    poll_interval_seconds: float = 2.0,
    poll_max_wait_seconds: float = 30.0,
) -> None:
    """
    Ensure there is enough free VRAM to load the model.

    If free VRAM is below *min_free_gib* and *auto_kill* is True, kills
    stale GPU processes and waits for VRAM to free up.
    """
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping VRAM check.")
        return

    free_gib = _free_vram_gib(device)
    total_gib = _gib(torch.cuda.get_device_properties(device).total_memory)
    log.info(
        "GPU %d: %.1f GiB free / %.1f GiB total (need %.1f GiB)",
        device, free_gib, total_gib, min_free_gib,
    )

    if free_gib >= min_free_gib:
        return

    other_pids = _get_other_gpu_pids(device)

    if not other_pids:
        raise RuntimeError(
            f"Insufficient VRAM: {free_gib:.1f} GiB free, {min_free_gib:.1f} GiB needed. "
            "No other compute processes detected — "
            "try setting PYTORCH_ALLOC_CONF=expandable_segments:True."
        )

    if not auto_kill:
        pid_str = " ".join(str(p) for p in other_pids)
        raise RuntimeError(
            f"Insufficient VRAM: {free_gib:.1f} GiB free, {min_free_gib:.1f} GiB needed. "
            f"Stale GPU processes: {other_pids}. "
            f"Kill them with: kill -9 {pid_str}"
        )

    log.warning(
        "Only %.1f GiB free (need %.1f GiB). Auto-killing stale GPU processes: %s",
        free_gib, min_free_gib, other_pids,
    )
    _kill_pids(other_pids)

    deadline = time.monotonic() + poll_max_wait_seconds
    while time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        release_gpu_memory()
        if _free_vram_gib(device) >= min_free_gib:
            log.info("VRAM freed. %.1f GiB available.", _free_vram_gib(device))
            return

    raise RuntimeError(
        f"Timed out waiting for VRAM. {_free_vram_gib(device):.1f} GiB free, "
        f"need {min_free_gib:.1f} GiB."
    )


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def _build_quantization_config(cfg: MaltModelConfig):
    if not cfg.load_in_4bit:
        return None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig not available. Install `bitsandbytes` or set "
            "load_in_4bit=False in MaltModelConfig."
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.torch_dtype,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_malt_llama_with_adapters(
    cfg: MaltModelConfig | None = None,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load the base model (GanitLLM-4B by default) and attach three LoRA
    adapters for the Generator, Verifier, and Refiner roles.

    Returns a PEFT-wrapped model and its tokenizer with three named
    adapters: "generator", "verifier", "refiner".
    """
    cfg = cfg or MaltModelConfig()

    release_gpu_memory()
    check_gpu_memory()

    quant_config = _build_quantization_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=cfg.device_map,
        quantization_config=quant_config,
        torch_dtype=cfg.torch_dtype if quant_config is None else None,
    )
    log.info("Base model loaded: %s", cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )

    peft_model = get_peft_model(model, lora_config, adapter_name=ROLE_GENERATOR)
    peft_model.add_adapter(ROLE_VERIFIER, lora_config)
    peft_model.add_adapter(ROLE_REFINER, lora_config)
    peft_model.set_adapter(ROLE_GENERATOR)

    log.info("LoRA adapters attached (generator / verifier / refiner)")

    return peft_model, tokenizer


def set_active_role_adapter(model: PeftModel, role: str) -> None:
    """Switch the active LoRA adapter on a shared base model."""
    if role not in (ROLE_GENERATOR, ROLE_VERIFIER, ROLE_REFINER):
        raise ValueError(f"Unknown role adapter: {role!r}")
    model.set_adapter(role)


def load_malt_llama_with_trained_adapters(
    cfg: MaltModelConfig | None = None,
    generator_checkpoint: str | Path | None = None,
    verifier_checkpoint: str | Path | None = None,
    refiner_checkpoint: str | Path | None = None,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load the base model with role adapters, then optionally load
    trained adapter weights from PEFT checkpoints on disk.
    """
    model, tokenizer = load_malt_llama_with_adapters(cfg)

    if verifier_checkpoint is not None:
        ckpt = Path(verifier_checkpoint)
        model.load_adapter(str((ckpt / ROLE_VERIFIER).resolve()), adapter_name=ROLE_VERIFIER)
    if generator_checkpoint is not None:
        ckpt = Path(generator_checkpoint)
        model.load_adapter(str((ckpt / ROLE_GENERATOR).resolve()), adapter_name=ROLE_GENERATOR)
    if refiner_checkpoint is not None:
        ckpt = Path(refiner_checkpoint)
        model.load_adapter(str((ckpt / ROLE_REFINER).resolve()), adapter_name=ROLE_REFINER)

    model.set_adapter(ROLE_GENERATOR)

    return model, tokenizer
