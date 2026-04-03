python eval_mgsm_sharded.py \
  --gen-ckpt checkpoints/generator_sft \
  --ver-ckpt checkpoints/verifier_dpo \
  --ref-ckpt checkpoints/refiner_dpo \
  --shard-size 128 \
  --chunk-size 8 \
  --num-samples 3

python eval_malt_new.py \
--devset data/mgsm_bn_single.tsv \
--output-dir output/eval/mgsm_bn_single \
--chunk-size 20 \
--num-samples 3 \
  --gen-checkpoint checkpoints/generator_sft \
  --ver-checkpoint checkpoints/verifier_dpo \
  --ref-checkpoint checkpoints/refiner_dpo 


python eval_mvsamp_sharded.py \
  --gen-ckpt checkpoints/generator_sft \
  --ver-ckpt checkpoints/verifier_dpo \
  --ref-ckpt checkpoints/refiner_dpo \
  --chunk-size 20 \
  --num-samples 3




python eval_malt.py \
--devset data/dev_rand10.csv \
--output-dir output/rand10_wo_lora \
  --gen-checkpoint checkpoints/generator_sft \
  --ver-checkpoint checkpoints/verifier_dpo \
  --ref-checkpoint checkpoints/refiner_dpo \
  --chunk-size 16 \
  --num-samples 3

python eval_malt.py \
--devset data/dev_rand10.csv \
--output-dir output/dev_rand10_new \
  --gen-checkpoint checkpoints/generator_sft \
  --ver-checkpoint checkpoints/verifier_dpo \
  --ref-checkpoint checkpoints/refiner_dpo \
  --chunk-size 16 \
  --num-samples 3


python evaluate_ganitllm.py \
  --csv data/msvamp_test.csv \
  --max_new_tokens 2048 \
  --temperature 0.7 \
  --output_json results.json