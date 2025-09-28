```sh
arbor serve --arbor-config arbor.yaml
```

```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique_dspy.py train \
    --model $MODEL \
    2>&1 | tee outputs/train-$(date +%s).log
```