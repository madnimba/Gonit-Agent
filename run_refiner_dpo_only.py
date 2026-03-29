from pathlib import Path
import gc
import torch
from malt.training.dpo_trainer import DpoTrainingConfig, train_refiner_dpo

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

valued_path = Path("data/all_trajectories.valued.jsonl")
r_dpo_cfg = DpoTrainingConfig(
    output_dir=Path("checkpoints/refiner_dpo"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
train_refiner_dpo(valued_path, r_dpo_cfg, task="somadhan")
cleanup()

