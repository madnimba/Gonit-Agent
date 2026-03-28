# Gonit-Agent (MALT multi-agent reasoning for math datasets)

This repository implements a full multi-agent math reasoning pipeline inspired by MALT. It is centered around a shared base model (default: GanitLLM 4B) with 3 LoRA adapters for:

- Generator (G)
- Verifier (V)
- Refiner (R)

The pipeline includes:

- Tree search: G→V→R trajectories with configurable branching factor.
- Value iteration: credit assignment from final refiner scores back to verifier/generator.
- SFT (supervised fine-tuning) for all roles.
- DPO (direct preference optimization) for verifier and refiner.
- Baselines with Qwen zero-shot.

Supported datasets:

- Somadhan (CSV in `data/SOMADHAN*.csv`)
- GSM8K (`malt.data.gsm8k` via HuggingFace)
- MATH (`malt.data.math` via HuggingFace)

---

## 1. Requirements

- Python 3.10+
- PyTorch + CUDA (GPU recommended; 24GB VRAM preferred for default model)
- `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes` (if using 4-bit)

Install:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate   # Linux/macOS

pip install --upgrade pip
pip install -r requirements.txt
```

### Notes

- Default model is `dipta007/GanitLLM-4B_SFT_CGRPO` (GanitLLM, Qwen3-derived).
- For low VRAM, set `MaltModelConfig(load_in_4bit=True)` and install `bitsandbytes`.
- On Windows, `bitsandbytes` builds may require WSL2 or specific CUDA versions.

---

## 2. Project structure

- `malt/models/`
  - `base_model.py` - model loading, LoRA adapters, adapter switching, GPU memory helpers.
  - `prompts.py` - prompt templates for G/V/R and chat formatting.
- `malt/data/`
  - `somadhan.py` - Somadhan CSV loader and answer normalization/matching.
  - `gsm8k.py` - GSM8K loader and answer normalization/matching.
  - `math.py` - MATH loader and answer normalization/matching.
  - `preference_builders.py` - trajectory -> SFT/DPO sample builders.
- `malt/search/`
  - `tree_search.py` - G→V→R trajectory generation, per-question progress, checkpointing.
  - `value_iteration.py` - trajectory value assignment and JSONL augmentation.
- `malt/training/`
  - `sft_trainer.py` - `SftTrainingConfig`, SFT training functions by role.
  - `dpo_trainer.py` - `DpoTrainingConfig`, DPO training functions by role.
- `malt/inference/`
  - `pipeline.py` - single-agent and multi-agent inference and majority voting.
  - `qwen_baseline.py` - zero-shot Qwen baseline for comparison.
- `malt/utils/`
  - `eval.py` - evaluation metrics (GSM8K, MATH, Somadhan).
- `scripts/`
  - `eval_malt.py` - evaluation harness for Somadhan (and trained checkpoints).
- `generate_trajectories.py` - helper for Somadhan trajectory generation with chunking and merging.
- `run.sh` - old convenience script (may require `connectomics_ctrl.py` which is outside this repo).

---

## 3. Data loaders

Somadhan:

```python
from malt.data import load_Somadhan_split
examples = load_Somadhan_split("data/SOMADHAN.csv", id_start=1)
```

GSM8K:

```python
from malt.data import load_gsm8k_split
examples = load_gsm8k_split("validation")
```

MATH:

```python
from malt.data import load_math_split
examples = load_math_split("test")
```

Answer extraction and matching helpers:

- `extract_Somadhan_answer`, `normalize_Somadhan_answer`, `Somadhan_exact_match`
- `extract_gsm8k_answer`, `normalize_gsm8k_answer`, `gsm8k_exact_match`
- `extract_math_answer`, `normalize_math_answer`, `math_exact_match`

---

## 4. Generate trajectories

For Somadhan (chunked, resumable):

```bash
python generate_trajectories.py \
  --csv-path data/SOMADHAN.csv \
  --output-dir output \
  --branching-factor 3 \
  --chunk-size 500 \
  --id-start 1
```

To merge chunks:

```bash
python generate_trajectories.py --merge-only --merge-dir output/trajectories --merge-output data/all_trajectories.jsonl
```

Alternatively, use API in code:

```python
from malt.search.tree_search import TreeSearchConfig, run_tree_search_for_somadhan
from malt.models import MaltModelConfig

cfg = TreeSearchConfig(branching_factor=3, output_path=Path("data/all_trajectories.jsonl"))
run_tree_search_for_somadhan("data/SOMADHAN.csv", cfg, model_cfg=MaltModelConfig())
```

For GSM8K:

```python
from malt.search.tree_search import TreeSearchConfig, run_tree_search_for_gsm8k_split
cfg = TreeSearchConfig(branching_factor=3, output_path=Path("data/gsm8k_trajectories.jsonl"))
run_tree_search_for_gsm8k_split("validation", cfg)
```

---

## 5. Value iteration

```python
from pathlib import Path
from malt.search.value_iteration import value_iteration_over_jsonl, ValueIterationConfig

input_path = Path("data/all_trajectories.jsonl")
valued_path = value_iteration_over_jsonl(input_path, cfg=ValueIterationConfig(task="somadhan", threshold=0.5))
```

This writes `data/all_trajectories.valued.jsonl` and attaches `value`/`label` to nodes.

---

## 6. Training

### SFT training

```python
from pathlib import Path
from malt.training.sft_trainer import SftTrainingConfig, train_generator_sft, train_verifier_sft, train_refiner_sft

sft_cfg = SftTrainingConfig(output_dir=Path("checkpoints/generator_sft"), num_train_epochs=3)
train_generator_sft(Path("data/all_trajectories.valued.jsonl"), sft_cfg, task="somadhan")

v_cfg = SftTrainingConfig(output_dir=Path("checkpoints/verifier_sft"))
train_verifier_sft(Path("data/all_trajectories.valued.jsonl"), v_cfg, task="somadhan")

r_cfg = SftTrainingConfig(output_dir=Path("checkpoints/refiner_sft"))
train_refiner_sft(Path("data/all_trajectories.valued.jsonl"), r_cfg, task="somadhan")
```

### DPO training

```python
from pathlib import Path
from malt.training.dpo_trainer import DpoTrainingConfig, train_verifier_dpo, train_refiner_dpo

v_dpo_cfg = DpoTrainingConfig(output_dir=Path("checkpoints/verifier_dpo"))
train_verifier_dpo(Path("data/all_trajectories.valued.jsonl"), v_dpo_cfg, task="somadhan")

r_dpo_cfg = DpoTrainingConfig(output_dir=Path("checkpoints/refiner_dpo"))
train_refiner_dpo(Path("data/all_trajectories.valued.jsonl"), r_dpo_cfg, task="somadhan")
```

The DPO runners load SFT adapter checkpoints from `checkpoints/verifier_sft` and `checkpoints/refiner_sft` by default.

---

## 7. Inference and evaluation

### Single-agent generator (MV)

```python
from malt.inference.pipeline import InferenceConfig, run_single_agent_generator
from malt.models import load_malt_llama_with_trained_adapters

model, tok = load_malt_llama_with_trained_adapters(
    generator_checkpoint="checkpoints/generator_sft",
    verifier_checkpoint="checkpoints/verifier_dpo",
    refiner_checkpoint="checkpoints/refiner_dpo",
)

cfg = InferenceConfig(num_samples=3, temperature=0.3)
questions = load_Somadhan_split("data/somadhan_dev.csv")
preds = run_single_agent_generator(model, tok, questions, cfg)
```

### Multi-agent MALT (G→V→R, MV)

```python
from malt.inference.pipeline import run_multi_agent_malt

preds = run_multi_agent_malt(model, model, model, tok, questions, cfg)
```

### Qwen zero-shot baseline

```python
from malt.inference.qwen_baseline import QwenBaselineConfig, load_qwen_model_and_tokenizer, run_qwen_zero_shot

qwen_cfg = QwenBaselineConfig(model_name="Qwen/Qwen2.5-1.5B", num_samples=1)
base_model, base_tok = load_qwen_model_and_tokenizer(qwen_cfg)
baseline_preds = run_qwen_zero_shot(base_model, base_tok, questions, qwen_cfg)
```

### Eval script

```bash
python scripts/eval_malt.py --devset data/somadhan_dev.csv --num-samples 3 \
  --gen-checkpoint checkpoints/generator_sft \
  --ver-checkpoint checkpoints/verifier_dpo \
  --ref-checkpoint checkpoints/refiner_dpo
```

This returns:
- GanitLLM base zero-shot
- GanitLLM base MV
- Trained generator-only MV
- MALT G→V→R MV

---

## 8. Notes

- The model uses a shared base with role adapters set by `set_active_role_adapter`.
- `TreeSearchConfig` has tuner for branch factor, max tokens, temperature, top-p/top-k, batch size, torch.compile.
- `value_iteration` supports `task` = `"somadhan" | "gsm8k" | "math"`.
- To integrate new datasets, implement question-class and answer exact-match functions in `malt/data`.
- For local experimentation, reduce `branching_factor` and `num_samples`.

---

## 9. Troubleshooting

- If GPU OOM in `tree_search` or inference: reduce batch size, `max_new_tokens`, or use 4-bit 
  quantization (`MaltModelConfig(load_in_4bit=True)`).
- If HF dataset download fails, check network and `HUGGING_FACE_HUB_TOKEN` (for private access).
- If `bitsandbytes` missing on Windows, try WSL2 or Linux container.

