## Environment Installation

```bash
# Install the MuSiQue environment package
vf-install vf-musique -p environments
```

## Quick Evaluation

```bash
vf-eval vf-musique --model meta-llama/Llama-3.1-8B-Instruct
```

## Training

Start vLLM inference server
```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1  --data-parallel-size 1 --enforce-eager --disable-log-requests --gpu-memory-utilization 0.7
```

Train on MuSiQue environment with GRPO
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train 
```


**Quick prediction:**
```bash
python scripts/train_musique.py predict \
    --model outputs/your-trained-model \
    --batch-size 8
```

## Evaluation of Trained Models

The evaluation script provides multiple commands for comprehensive analysis:

**Basic evaluation:**
```bash
python scripts/evaluate_musique.py outputs/your-trained-model
```

**Detailed evaluation:**
```bash
python scripts/evaluate_musique.py evaluate \
    outputs/your-trained-model \
    --dataset-split validation \
    --num-examples 100 \
    --retriever hybrid \
    --verbose
```
