Start vLLM inference server
```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-3B-Instruct \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enforce-eager \
    --disable-log-requests
```

## MuSiQue

Train on MuSiQue dataset
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
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
vf-install vf-math-python -p ./tmp/verifiers/environments
```

Train math dataset with Python tool
```sh
CUDA_VISIBLE_DEVICES=3 python scripts/train_math_python.py
```