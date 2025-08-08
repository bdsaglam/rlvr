## MuSiQue

Start vLLM inference server
```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --data-parallel-size 1 --enforce-eager --disable-log-requests
```

Install environment
```sh
vf-install vf-musique -p ./environments
```

Train on MuSiQue dataset
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train 
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