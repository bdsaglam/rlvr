## Commands

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 1 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 1 \
    --batch-size 6 \
    --num-generations 6 \
    --gradient-accumulation-steps 16 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 1 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250407_134926' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 1 \
    --batch-size 6 \
    --num-generations 6 \
    --gradient-accumulation-steps 8 \
    --resume-from-checkpoint \
    2>&1 | tee tmp/logs/train-$(date +%s).log


```sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --datasets 'bdsaglam/musique,answerable,train' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    2>&1 | tee tmp/ragent-llama3-8b-round-2-$(date +%s).log


accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250329_222430' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    --resume-from-checkpoint \
    2>&1 | tee tmp/ragent-llama3-8b-round-2-$(date +%s).log

```

```sh
export CUDA_VISIBLE_DEVICES=0,1

python scripts/ragent2.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    2>&1 | tee tmp/logs/train-$(date +%s).log

```

# 2025-04-11

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

# 2025-04-17

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --noise-rate 1.0 \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.5 \
    --n-env-jobs 32 \
    --batch-size 24 \
    --num-generations 12 \
    --gradient-accumulation-steps 8 \
    --lora-r 32 \
    --lora-alpha 64 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


# 2025-04-20

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 24 \
    --batch-size 24 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --n-epochs 4 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

python scripts/merge.py \
    outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250421_000014/checkpoint-400 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-400

python scripts/merge.py \
    outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250421_000014/checkpoint-1600 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1600

python scripts/merge.py \
    outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250421_000014/checkpoint-1900 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1900

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'Qwen/Qwen2.5-14B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --n-epochs 2 \
    2>&1 | tee tmp/logs/train-$(date +%s).log



## 2025-04-27

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --n-epochs 4 \
    --lora-r 64 \
    --lora-alpha 64 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

python scripts/merge.py outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250427_095331 --out outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250427_095331-merged

## 2025-05-05

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --n-epochs 2 \
    --lora-r 256 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-20250505_172203' \
    --resume-from-checkpoint \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --n-epochs 4 \
    --lora-r 128 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


## 2025-05-08

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --n-epochs 2 \
    --lora-r 256 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215' \
    --resume-from-checkpoint outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215/checkpoint-1600 \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --n-epochs 2 \
    --lora-r 256 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

python scripts/merge.py \
    ./outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215-merged

huggingface-cli upload --repo-type model \
    Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215-merged \
    ./outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215-merged




accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --n-epochs 1 \
    --lora-r 256 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 2 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 18 \
    --batch-size 18 \
    --num-generations 6 \
    --gradient-accumulation-steps 8 \
    --n-epochs 1 \
    --lora-r 256 \
    --lora-alpha 256 \
    --kl-beta 0.01 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


python scripts/merge.py \
    ./outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged

huggingface-cli upload --repo-type model \
    Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged \
    ./outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged


## 2025-05-22

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 24 \
    --batch-size 24 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --n-epochs 3 \
    --lora-r 256 \
    --lora-alpha 256 \
    --kl-beta 0.01 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


## Test set predictions

CUDA_VISIBLE_DEVICES=2 python scripts/ragent.py predict \
    --n-env-jobs 32 \
    --batch-size 32 \
    --model '/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged' \
    --temperature 0.5 \
    --top-p 0.95 \
    --few-shot-prob 0.0 \
    --dataset-path 'bdsaglam/musique-ans-test' \
    --dataset-name 'answerable' \
    --dataset-split 'test' \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --out tmp/musique-test/predictions.jsonl


## 2025-05-23

Resume training the best fine-tuned model.

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model '/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --n-epochs 1 \
    --lora-r 64 \
    --lora-alpha 64 \
    --kl-beta 0.01 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

## Download then Merge

huggingface-cli download bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809

python scripts/merge.py \
    /home/baris/.cache/huggingface/hub/models--bdsaglam--Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809/snapshots/881c06f8f8143a55e1f4a975544ec4324f107c68 \
    --out ./tmp/outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged

## Resume training

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --n-epochs 1 \
    --lora-r 64 \
    --lora-alpha 32 \
    --kl-beta 0.01 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

## 2025-05-26

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0.01 \
    --n-env-jobs 8 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --n-epochs 2 \
    --no-peft \
    2>&1 | tee tmp/logs/train-$(date +%s).log

## 2025-05-28

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train;bdsaglam/hotpotqa-distractor,default,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0.01 \
    --n-env-jobs 8 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --n-epochs 1 \
    --no-peft \
    2>&1 | tee tmp/logs/train-$(date +%s).log

## 2025-05-28

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'Qwen/Qwen2.5-14B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0.01 \
    --n-env-jobs 8 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --n-epochs 1 \
    --lora-r 512 \
    --lora-alpha 512 \
    2>&1 | tee tmp/logs/train-$(date +%s).log



## 2025-05-30

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'Qwen/Qwen2.5-14B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0.01 \
    --n-env-jobs 8 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --n-epochs 1 \
    --lora-r 512 \
    --lora-alpha 512 \
    --scale-rewards \
    2>&1 | tee tmp/logs/train-$(date +%s).log


python scripts/merge.py \
    ./outputs/Qwen2.5-14B-Instruct-ragent-grpo-20250530_155020/checkpoint-200 \
    --out outputs/Qwen2.5-14B-Instruct-ragent-grpo-20250530_155020-merged

huggingface-cli upload --repo-type model \
    Qwen2.5-14B-Instruct-ragent-grpo-20250530_155020-merged \
    ./outputs/Qwen2.5-14B-Instruct-ragent-grpo-20250530_155020-merged


## 2025-05-31

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0.02 \
    --n-env-jobs 8 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --n-epochs 1 \
    --no-peft \
    --scale-rewards \
    2>&1 | tee tmp/logs/train-$(date +%s).log

## 2025-06-01

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0.04 \
    --n-env-jobs 16 \
    --batch-size 16 \
    --num-generations 16 \
    --gradient-accumulation-steps 16 \
    --n-epochs 1 \
    --no-peft \
    --scale-rewards \
    2>&1 | tee tmp/logs/train-$(date +%s).log

### Dr. GRPO recipe

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 1 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0 \
    --n-env-jobs 16 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --n-epochs 1 \
    --lora-r 256 \
    --lora-alpha 256 \
    --max-completion-length 2048 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

python scripts/merge.py \
    ./outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250602_094840/checkpoint-200 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250602_094840-merged

## 2025-06-03

python scripts/ragent.py predict \
      --n-env-jobs 32 \
      --batch-size 32 \
      --n-env-jobs 1 \
      --model openai/gpt-4.1 \
      --temperature 0.5 \
      --top-p 0.95 \
      --few-shot-prob 0.0 \
      --dataset-path bdsaglam/musique-mini \
      --dataset-name answerable \
      --dataset-split validation \
      --retriever hybrid-tei \
      --retriever-top-k 1 \
      --repeat 1 \
      --out tmp/gpt/predictions.jsonl
    
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 1 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --kl-beta 0 \
    --n-env-jobs 16 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 16 \
    --n-epochs 1 \
    --lora-r 512 \
    --lora-alpha 512 \
    --max-completion-length 2048 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

## 2025-06-05

huggingface-cli download bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250603_205328 --repo-type model

python scripts/merge.py \
    bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250603_205328 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-20250603_205328-merged

huggingface-cli upload --repo-type model \
    bdsaglam/Llama-3.1-8B-Instruct-ragent-20250603_205328-merged \
    ./outputs/Llama-3.1-8B-Instruct-ragent-20250603_205328-merged/

conda install -c conda-forge cudatoolkit-dev -y
