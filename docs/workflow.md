## Setup

```sh
docker-compose down --remove-orphans; docker-compose up --build
```

## Train

```sh
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --repo-type model
```

```sh
export CUDA_VISIBLE_DEVICES=0,1,2
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 2 \
    scripts/ragent.py train \
    2>&1 | tee tmp/logs/train-$(date +%s).log
```

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --datasets 'bdsaglam/musique,answerable,train;bdsaglam/hotpotqa-distractor,default,train' \
    --few-shot-prob 1.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 1 \
    2>&1 | tee tmp/logs/train-$(date +%s).log
```

### Publish manually
```sh
huggingface-cli upload --repo-type model \
    Qwen2.5-1.5B-Instruct-ragent-grpo-musique \
    ./outputs/Qwen2.5-1.5B-Instruct-ragent-grpo-musique
```


### Resume training

```sh
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250329_222430' \
    --few-shot-prob 1.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    --resume-from-checkpoint \
    2>&1 | tee tmp/logs/train-$(date +%s).log

```

## Predict

```sh
python scripts/merge.py \
    ./outputs/Qwen2.5-1.5B-Instruct-ragent-grpo-musique/checkpoint-1000 \
    --out outputs/Qwen2.5-1.5B-Instruct-ragent-grpo-musique-merged
```

```sh
# export MODEL=openai/meta-llama/Llama-3.3-70B-Instruct-Turbo
# export MODEL=Qwen/Qwen2.5-32B-Instruct
# export MODEL=meta-llama/Llama-3.1-8B-Instruct
export MODEL=bdsaglam/Qwen2.5-1.5B-Instruct-ragent-musique
export RETRIEVER=hybrid
export RETRIEVER_top_n=2
python scripts/ragent.py predict \
    --model $MODEL \
    --dataset-path bdsaglam/musique-mini \
    --dataset-name answerable \
    --dataset-split validation \
    --retriever $RETRIEVER \
    --retriever-top-k $RETRIEVER_top_n \
    --n-env-jobs 32 \
    --batch-size 32 \
    --out outputs/ragent/$MODEL/predictions-musique-mini-$RETRIEVER-$RETRIEVER_top_n.jsonl
```


## Rerank

```sh
curl 127.0.0.1:8930/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Neural networks are a type of machine learning model.", "Symbolic AI is a type of machine learning model."]}' \
    -H 'Content-Type: application/json'

curl http://localhost:8931/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Neural networks are a type of machine learning model.", "Symbolic AI is a type of machine learning model."]}' \
    -H 'Content-Type: application/json'

curl http://localhost:8931/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Neural networks are a type of machine learning model.", "Symbolic AI is a type of machine learning model."], "model": "tei"}' \
    -H 'Content-Type: application/json'
```
