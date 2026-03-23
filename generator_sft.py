"""Train the Generator LoRA adapter (SFT only, no DPO — per MALT paper)."""

from pathlib import Path
from malt.training.sft_trainer import SftTrainingConfig, train_generator_sft

sft_cfg = SftTrainingConfig(
    output_dir=Path("checkpoints/generator_sft"),
    num_train_epochs=3,
)
train_generator_sft(
    Path("data/all_trajectories.valued.jsonl"),
    sft_cfg,
    task="somadhan",
)
