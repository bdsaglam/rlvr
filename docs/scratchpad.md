Start necessary services
```sh
docker-compose down --remove-orphans; docker-compose up --build
```

Start vLLM inference server with Qwen2.5 7B.
```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager

```

Start vLLM inference server with Llama3.1 8B. Llama3.1 tokenizer config includes tool calling template but we should use the custom one provided tailored to vLLM.

```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template services/vllm/tool_chat_template_llama3.1_json.jinja \
    --enforce-eager \
    --disable-log-requests
```

## MuSiQue

Train on MuSiQue dataset
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model Qwen/Qwen2.5-7B-Instruct \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-completion-length 1024 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```

Evaluate the model on the validation set
```sh
python scripts/musique.py evaluate \
    --datasets "bdsaglam/musique-mini,answerable,validation" \
    --model Qwen/Qwen2.5-7B-Instruct 
```


## GSM8K

Install environment
```sh
vf-install vf-gsm8k -p ./tmp/verifiers/environments
```

Train gsm8k with GRPO
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    ./tmp/verifiers/examples/grpo/train_gsm8k.py
```

## Math with Python tool

Install environment
```sh
vf-install math-python --from-repo
```

```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```

Train math dataset with Python tool
```sh
CUDA_VISIBLE_DEVICES=3 python scripts/train_math_python.py
```

```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_math_python.py \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```

## 2025-08-12

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-completion-length 1024 \
    --lora-r 64 \
    --lora-alpha 64 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log


CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-completion-length 1024 \
    --no-peft \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log


CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --no-peft \
    --max-completion-length 1024 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log


CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model Qwen/Qwen2.5-14B-Instruct \
    --lora-r 64 \
    --lora-alpha 64 \
    --max-completion-length 1024 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    2>&1 | tee outputs/logs/train-$(date +%s).log


CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model Qwen/Qwen2.5-14B-Instruct \
    --lora-r 64 \
    --lora-alpha 64 \
    --max-completion-length 1024 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    2>&1 | tee outputs/logs/train-$(date +%s).log


vf-eval vf-musique --model Qwen/Qwen2.5-3B-Instruct --api-base-url http://0.0.0.0:8000/v1 --api-key local


CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-completion-length 1024 \
    --scale-rewards \
    --loss-type dr_grpo \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    2>&1 | tee outputs/logs/train-$(date +%s).log

# Without DeepSpeed ZeRO (model replication instead of sharding)
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train"  \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-completion-length 1024 \
    --scale-rewards \
    --loss-type dr_grpo \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    2>&1 | tee outputs/logs/train-$(date +%s).log

# 

Inference
```sh
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --data-parallel-size 2 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```

Training
```sh
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num-processes 2 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train[:256]"  \
    --model Qwen/Qwen2.5-7B-Instruct \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```

#  Llama 3.1 8B

It doesn't work without custom chat template.
```sh
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --data-parallel-size 2 \
    --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template services/vllm/tool_chat_template_llama3.1_json.jinja \
    --enforce-eager
```

Training
```sh
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num-processes 2 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique,answerable,train[:256]"  \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```


Raises "ValueError: This model only supports single tool-calls at once!"

# Qwen2.5-7B No PEFT

Inference
```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"
export TOOL_CALL_PARSER="hermes"

CUDA_VISIBLE_DEVICES=0 vf-vllm \
    --model $MODEL \
    --enable-auto-tool-choice  --tool-call-parser $TOOL_CALL_PARSER \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --port 8000 \
    --enforce-eager
```

Training
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train"  \
    --model $MODEL \
    --no-peft \
    --loss-type "dr_grpo" \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```
Raises OOM error.

# Qwen3-8B

```sh
export MODEL="Qwen/Qwen3-8B"
export TOOL_CALL_PARSER="hermes"
```

Inference
```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER \
    --enforce-eager
```

Training
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train"  \
    --model $MODEL \
    --no-peft \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```

# Qwen2.5-7B PEFT

Inference
```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"
export TOOL_CALL_PARSER="hermes"

CUDA_VISIBLE_DEVICES=0 vf-vllm \
    --model $MODEL \
    --enable-auto-tool-choice  --tool-call-parser $TOOL_CALL_PARSER \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --port 8000 \
    --enforce-eager
```

Training
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train"  \
    --model $MODEL \
    --lora-r 64 \
    --lora-alpha 64 \
    --loss-type "dr_grpo" \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```
Reward stalls.


```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train"  \
    --model $MODEL \
    --loss-type "dr_grpo" \
    --kl-beta 0 \
    --no-peft \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```


```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train"  \
    --model $MODEL \
    --loss-type "grpo" \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --bf16 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```

# MuSiQue stable training
https://wandb.ai/bdsaglam/rlvr-debug/runs/ja1sqr1c

Install environment

```sh
vf-install vf_musique 
```


Inference 

```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
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
    --bf16 \
    --loss-type "grpo" \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --scale-rewards \
    --max-grad-norm 0.01 \
    --learning-rate 1e-5 \
    2>&1 | tee outputs/train-$(date +%s).log
```


export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager



export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique-mini,answerable,train" \
    --noise 0 \
    --model $MODEL \
    --temperature 0.7 \
    --min-p 0.1 \
    --lora-r 16 \
    --lora-alpha 16 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log

## Hyperparameter exploration
Params to explore: temperature, min-p, lora-r, lora-alpha, max-grad-norm, learning-rate


export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique-mini,answerable,train[:16]" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --max-steps 100 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --min-p 0.1 \
    --lora-r 16 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log

## temperature and min-p

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique-mini,answerable,train[:16]" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --max-steps 100 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --min-p 0.05 \
    --lora-r 16 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique-mini,answerable,train[:16]" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --max-steps 100 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 1.0 \
    --min-p 0.1 \
    --lora-r 16 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 16 \
    --max-grad-norm 0.01 \
    --learning-rate 1e-5 \
    2>&1 | tee outputs/train-$(date +%s).log

## musique-multi

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique-multi \
    --datasets "bdsaglam/musique-mini,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 16 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log


OPENAI_API_KEY=local uv run vf-eval xml-tool-env \
  -m Qwen/Qwen2.5-7B-Instruct \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"dataset_name": "math", "split": "train"}' \
  --api-base-url http://0.0.0.0:8000/v1

## musique train split

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 4 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 16 \
    --max-grad-norm 0.01 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 32 \
    --max-grad-norm 0.01 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 16 \
    --max-grad-norm 0.01 \
    --learning-rate 1e-6 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 16 \
    --max-grad-norm 0.01 \
    --learning-rate 5e-6 \
    --num-iterations 2 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 64 \
    --lora-alpha 16 \
    --max-grad-norm 0.01 \
    --learning-rate 1e-6 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 64 \
    --lora-alpha 16 \
    --max-grad-norm 0.01 \
    --learning-rate 1e-6 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 12 \
    --gradient-accumulation-steps 4 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log

## Mistral


export MODEL="mistralai/Mistral-7B-Instruct-v0.3"
CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --chat-template patches/vllm/tool_chat_template_mistral_parallel.jinja \
    --enforce-eager

export MODEL="mistralai/Mistral-7B-Instruct-v0.3"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 0 \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 8 \
    --num-generations 12 \
    --gradient-accumulation-steps 4 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log



#

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 4 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 64 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log


export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=3 python scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 1.0 \
    --kl-beta 0.00 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 1 \
    --num-generations 16 \
    --gradient-accumulation-steps 16 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 16 \
    --lora-alpha 64 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-5 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=3 python scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --noise 1.0 \
    --kl-beta 0.00 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 1 \
    --num-generations 16 \
    --gradient-accumulation-steps 16 \
    --model $MODEL \
    --temperature 0.6 \
    --lora-r 64 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-5 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log

## Thinking Machines LoRA recipe

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm --model $MODEL \
    --port 8007 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=3 python scripts/train_musique.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --kl-beta 0.00 \
    --loss-type "dr_grpo" \
    --bf16 \
    --batch-size 1 \
    --num-generations 16 \
    --gradient-accumulation-steps 16 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 128 \
    --lora-alpha 16 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-6 \
    --num-iterations 1 \
    2>&1 | tee outputs/logs/train-$(date +%s).log


# Unsloth

export MODEL="unsloth/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype float16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --data-parallel-size 3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager

export MODEL="unsloth/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=3 python scripts/train_musique_unsloth.py train \
    --env-id vf-musique \
    --datasets "bdsaglam/musique,answerable,train" \
    --kl-beta 0.0 \
    --loss-type "dr_grpo" \
    --dtype float16 \
    --batch-size 1 \
    --num-generations 12 \
    --gradient-accumulation-steps 12 \
    --model $MODEL \
    --temperature 0.7 \
    --lora-r 32 \
    --lora-alpha 64 \
    --max-grad-norm 0.1 \
    --learning-rate 5e-6 \
    --num-iterations 1 \
    2>&1 | tee outputs/train-$(date +%s).log


CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8007 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager

prime eval run vf-musique -m Qwen/Qwen2.5-7B-Instruct -b http://0.0.0.0:8007/v1

prime eval run arc-agi -m Qwen/Qwen2.5-7B-Instruct -b http://0.0.0.0:8007/v1

prime eval run arc-agi -x '{"data_dir":"data/arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen2.5-7B-Instruct -b http://0.0.0.0:8007/v1


CUDA_VISIBLE_DEVICES=0,1,2 vllm serve mit-oasys/rlm-qwen3-8b-v0.1 \
    --port 8007 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enforce-eager

prime eval run arc-agi -m mit-oasys/rlm-qwen3-8b-v0.1 -b http://0.0.0.0:8007/v1

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8007 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-120b \
    --port 8007 \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser openai \
    --enforce-eager

prime eval run arc-agi -x '{"data_dir":"data/arc-dummy"}' -n 1 -m openai/gpt-oss-120b -b http://0.0.0.0:8007/v1

CUDA_VISIBLE_DEVICES=0,1 vllm serve openai/gpt-oss-20b \
    --port 8007 \
    --async-scheduling \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser openai \
    --enforce-eager

prime eval run arc-agi -x '{"data_dir":"data/arc-dummy"}' -n 1 -m openai/gpt-oss-20b -b http://0.0.0.0:8007/v1

uv run rl @ configs/prime-rl/arc-agi.toml

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --port 8007 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enforce-eager