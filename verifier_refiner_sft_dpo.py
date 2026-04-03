"""
Train Verifier and Refiner LoRA adapters (SFT then DPO for each).

Runs four training stages sequentially, clearing GPU memory between each:
  1. Verifier SFT
  2. Refiner  SFT
  3. Verifier DPO  (loads verifier SFT checkpoint)
  4. Refiner  DPO  (loads refiner  SFT checkpoint)
"""

from pathlib import Path
import gc
import torch

from malt.training.sft_trainer import SftTrainingConfig, train_verifier_and_refiner_sft
from malt.training.dpo_trainer import DpoTrainingConfig, train_verifier_dpo, train_refiner_dpo


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


valued_path = Path("data/all_trajectories.valued.jsonl")

v_sft_cfg = SftTrainingConfig(
    output_dir=Path("checkpoints/verifier_sft"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
r_sft_cfg = SftTrainingConfig(
    output_dir=Path("checkpoints/refiner_sft"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)

train_verifier_and_refiner_sft(valued_path, v_sft_cfg, r_sft_cfg, task="somadhan")
cleanup()

v_dpo_cfg = DpoTrainingConfig(
    output_dir=Path("checkpoints/verifier_dpo"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
r_dpo_cfg = DpoTrainingConfig(
    output_dir=Path("checkpoints/refiner_dpo"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)

train_verifier_dpo(valued_path, v_dpo_cfg, task="somadhan")
cleanup()

train_refiner_dpo(valued_path, r_dpo_cfg, task="somadhan")
cleanup()
