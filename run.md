# Commands to run this repo

From the repository root, use `PYTHONPATH=.` (Unix) or set `PYTHONPATH` to `.` on Windows so the `malt` package resolves. Examples below use Unix-style env; on PowerShell: `$env:PYTHONPATH = "."`.

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows: `venv\Scripts\activate`.

### Install requirements

```bash
pip install -r requirements.txt
```

---

### Step 1: Generate trajectories (training data)

`generate_trajectories.py` runs MALT tree search on a Somadhan-style dataset.

| Option | Default | Description |
|--------|---------|-------------|
| `--csv-path` | `data/SOMADHAN.csv` | Path to the input CSV (Somadhan `question` / `answer` columns). |
| `--output-dir` | `output` | Root output directory; trajectories go under `output/trajectories/`. |
| `--id-start` | `1` | First example ID for this chunk (e.g. `1001` for chunk 2). |
| `--branching-factor` | `3` | Branching factor *n* (produces *n^3* trajectories per question). |
| `--chunk-size` | `500` | Rows per output JSONL chunk file. |
| `--merge-only` | off | Skip generation; only merge existing JSONL files. |
| `--merge-dir` | `<output-dir>/trajectories` | Directory of JSONL files to merge when using `--merge-only`. |
| `--merge-output` | `data/all_trajectories.jsonl` | Output path for merged JSONL. |

Example:

```bash
python generate_trajectories.py \
  --csv-path path/to/dataset.csv \
  --output-dir output \
  --branching-factor 3 \
  --chunk-size 500 \
  --id-start 1
```

### Merge chunks (manual)

```bash
cat chunk1.jsonl chunk2.jsonl ... > data/all_trajectories.jsonl
```

Or rely on `generate_trajectories.py --merge-only` (see table above).

---

### Step 2: Value iteration

No CLI flags. Reads `data/all_trajectories.jsonl` (or auto-merges from `output/trajectories/` if the merged file is missing).

```bash
python value_iterate.py
```

Produces `data/all_trajectories.valued.jsonl` (used by training).

---

### Step 3: Training

No CLI flags; paths and hyperparameters are set inside `train_all.py`.

```bash
python train_all.py
```

---

### Step 4: Evaluate

Run from repo root with `malt` on the path. Both scripts require `--output-dir` (created if missing) and write **`eval_results.txt`** there (human-readable summary plus `devset` / `num_samples` header). **`eval_with_ans.py`** also writes **`eval_predictions.jsonl`** in that directory (one JSON object per example: `question`, `ground_truth`, per-phase raw output, extracted answer, `answer_correctness`).

**Phases:** (1) GanitLLM zero-shot, (2) GanitLLM majority-vote on the base generator (uses `--num-samples`). If **all three** checkpoints are set, phases (3)–(6) run as MALT with **one chain per example** (no MV): untrained generator + trained V/R; trained G/R + untrained verifier; trained G/V + untrained refiner; fully trained G→V→R. If only some checkpoints are passed, phases 3–6 are skipped (with a note on stderr).

| Option | Default | Description |
|--------|---------|-------------|
| `--devset` | *(required)* | Path to dev data: CSV with `question` and `answer`, or `m_query` and `response` (e.g. `data/bn-msvamp.csv`), or `.tsv` with two tab-separated columns per line (e.g. `data/mgsm_bn.tsv`). |
| `--output-dir` | *(required)* | Directory for `eval_results.txt` (`eval_with_ans.py` also writes `eval_predictions.jsonl`). |
| `--num-samples` | `3` | Majority-vote sample count for **phase 2 only** (MV@*k*). |
| `--gen-checkpoint` | off | Generator adapter checkpoint (optional; all three needed for phases 3–6). |
| `--ver-checkpoint` | off | Verifier adapter checkpoint (optional). |
| `--ref-checkpoint` | off | Refiner adapter checkpoint (optional). |
| `--resume` | off | Skip phases whose cached predictions already exist in `--output-dir`. |
| `--quiet` | off | Disable tqdm per-example bars and per-phase ETA lines. |

**`eval_malt.py`** — prints the summary to stdout and writes it to `--output-dir/eval_results.txt`.

```bash
PYTHONPATH=. python eval_malt.py --devset data/somadhan_dev.csv --output-dir runs/eval

PYTHONPATH=. python eval_malt.py --devset data/mgsm_bn.tsv --output-dir runs/mgsm
PYTHONPATH=. python eval_malt.py --devset data/bn-msvamp.csv --output-dir runs/msvamp

PYTHONPATH=. python eval_malt.py --devset data/somadhan_dev.csv --output-dir runs/eval \
  --gen-checkpoint checkpoints/generator_sft \
  --ver-checkpoint checkpoints/verifier_dpo \
  --ref-checkpoint checkpoints/refiner_dpo

PYTHONPATH=. python eval_malt.py --devset data/dev.csv --output-dir runs/eval --quiet
PYTHONPATH=. python eval_malt.py --devset data/dev.csv --output-dir runs/eval --num-samples 5

# Resume after crash (skips any cached phases):
PYTHONPATH=. python eval_malt.py --devset data/dev.csv --output-dir runs/eval --resume
```

**`eval_with_ans.py`** — same as above, and always writes `eval_predictions.jsonl` under `--output-dir`.

```bash
PYTHONPATH=. python eval_with_ans.py --devset data/dev.csv --output-dir runs/eval \
  --gen-checkpoint checkpoints/generator_sft \
  --ver-checkpoint checkpoints/verifier_dpo \
  --ref-checkpoint checkpoints/refiner_dpo

# Resume after crash (skips any cached phases):
PYTHONPATH=. python eval_with_ans.py --devset data/dev.csv --output-dir runs/eval --resume
```

Without `--quiet`, evaluation shows tqdm progress bars (per example, with ETA) and a short line after each phase with elapsed time and an estimated time remaining for later phases.
