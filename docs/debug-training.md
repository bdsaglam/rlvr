# GSM8K

Install environment

```sh
vf-install gsm8k
```


Inference 

```sh
export MODEL="Qwen/Qwen2.5-3B-Instruct"

CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --enforce-eager
```


```sh
export MODEL="Qwen/Qwen2.5-3B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_gsm8k.py train \
    --model $MODEL \
    --loss-type "grpo" \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/train-$(date +%s).log
```

# MuSiQue

Install environment

```sh
vf-install vf_musique 
```


Inference 

```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```


```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --model $MODEL \
    --loss-type "grpo" \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --scale-rewards \
    --max-grad-norm 0.5 \
    --learning-rate 2e-5 \
    --bf16 \
    2>&1 | tee outputs/train-$(date +%s).log
```