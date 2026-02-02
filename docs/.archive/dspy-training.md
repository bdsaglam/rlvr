arbor serve --arbor-config arbor.yaml

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=3 python scripts/train_musique_dspy2.py train \
    --model $MODEL \
    2>&1 | tee outputs/logs/dspy-rlvr-train-$(date +%s).log