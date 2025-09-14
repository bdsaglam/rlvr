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
    2>&1 | tee outputs/logs/train-$(date +%s).log
```

# MuSiQue

Install environment

```sh
vf-install vf_musique 
```

## Qwen2.5-7B


Inference 

```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```


```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --structured \
    --model $MODEL \
    --bf16 \
    --loss-type "dr_grpo" \
    --kl-beta 0 \
    --lora-r 32 \
    --lora-alpha 64 \
    --batch-size 4 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log
```


```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --model $MODEL \
    --bf16 \
    --loss-type "dr_grpo" \
    --no-peft \
    --batch-size 2 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log
```

## Qwen3-4B

Inference 

```sh
export MODEL="willcb/Qwen3-4B"

CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```


```sh
export MODEL="willcb/Qwen3-4B"

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num-processes 2 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --model $MODEL \
    --bf16 \
    --loss-type "dr_grpo" \
    --scale-rewards \
    --lora-r 8 \
    --lora-alpha 16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-5 \
    2>&1 | tee outputs/train-$(date +%s).log
```

## Qwen3-8B

Inference 

```sh
export MODEL="willcb/Qwen3-8B"

CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager \
    2>&1 | tee outputs/logs/vllm-$(date +%s).log
```

Training

```sh
export MODEL="willcb/Qwen3-8B"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --model $MODEL \
    --max-prompt-length 8192 \
    --max-new-tokens 2048 \
    --bf16 \
    --loss-type "grpo" \
    --scale-rewards \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 4 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --max-grad-norm 0.01 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```


## Debug MuSiQue Training

```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```

```json
{
    "name": "Train MuSiQue",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/scripts/train_musique.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "train",
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--loss-type",
        "dr_grpo",
        "--batch-size",
        "1",
        "--num-generations",
        "2",
        "--gradient-accumulation-steps",
        "2"
    ],
    "env": {
        "CUDA_VISIBLE_DEVICES": "3",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_NET_GDR_DISABLE": "1",
        "NCCL_TREE_THRESHOLD": "0",
        "NCCL_ALGO": "Ring",
    }
}
```


```
OPENAI_API_KEY=local uv run vf-eval tool-test \
  -m Qwen/Qwen2.5-7B-Instruct \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 1000, "num_eval_examples": 100}' \
  --api-base-url http://0.0.0.0:8000/v1
```

Qwen2.5-7B-Instruct achieves perfect score (1.0) on tool-test eval with 100 examples.