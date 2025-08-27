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
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-3B-Instruct \
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

Inference
```sh
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --data-parallel-size 2 \
    --enable-auto-tool-choice --tool-call-parser llama3_json \
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