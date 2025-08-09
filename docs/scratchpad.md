## MuSiQue

Start vLLM inference server
```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-3B-Instruct \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enforce-eager \
    --disable-log-requests
```

Install environment
```sh
vf-install vf-musique -p ./environments
```

Train on MuSiQue dataset
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --datasets "bdsaglam/musique,answerable,train[:1000]" 
```


## GSM8K

Install environment
```sh
vf-install vf-gsm8k -p ./tmp/verifiers/environments
```

Start vLLM inference server
```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-3B-Instruct \
      --tensor-parallel-size 1  --data-parallel-size 2 --gpu-memory-utilization 0.7 \
      --enforce-eager --disable-log-requests
```

Train gsm8k with GRPO
```sh
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml \
    ./tmp/verifiers/examples/grpo/train_gsm8k.py
```
