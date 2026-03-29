# Commands to run this repo

### create a virtual environment
python -m venv venv
source venv/bin/activate

### install requirements
pip install -r requirements.txt

### Step 1: Generate trajectories to form training dataset
python generate_trajectories.py \
  --csv-path path/to/dataset \
  --output-dir output \
  --branching-factor 3 \
  --chunk-size 500 \
  --id-start 1

### Merge chunks
cat chunk1.jsonl chunk2.jsonl ... > data/all_trajectories.jsonl

### Step 2: Value iteration
python value_iterate.py

### Step 3: Training
python train_all.py

### Step 4: Evaluate
python eval_malt.py --devset data/somadhan_dev.csv \\
      --gen-checkpoint checkpoints/generator_sft \\
      --ver-checkpoint checkpoints/verifier_dpo \\
      --ref-checkpoint checkpoints/refiner_dpo

### Alternately, to save the answers along with accuracy scores
python eval_with_ans.py --devset data/dev.csv \\
      --dump-predictions output/eval_predictions.jsonl \\
      --gen-checkpoint checkpoints/generator_sft \\
      --ver-checkpoint checkpoints/verifier_dpo \\
      --ref-checkpoint checkpoints/refiner_dpo