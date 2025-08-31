# Training MuSiQue with DSPy

This guide explains how to use the DSPy-based training script for MuSiQue multi-hop question answering.

## Overview

The `train_musique_dspy.py` script implements multi-hop question answering training using DSPy's GRPO optimizer. Unlike the verifiers-based approach, this uses DSPy's native multi-hop reasoning patterns with the existing MuSiQue retrieval infrastructure.

## Prerequisites

1. **Start Arbor RL server** for model training:
```bash
# Create arbor.yaml configuration
cat > arbor.yaml << EOF
inference:
  gpu_ids: [0]
training:
  gpu_ids: [1, 2]
EOF

# Start Arbor server
python -m arbor.cli serve --arbor-config arbor.yaml
```

2. **Start rerank service** (for semantic/hybrid retrieval):
```bash
docker-compose up rerank
```

3. **Install vf_musique environment**:
```bash
cd environments/vf_musique && pip install -e .
```

## Usage

### Basic Training

Train with default settings (hybrid retrieval, 2 hops):

```bash
python scripts/train_musique_dspy.py train \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --train-size 600 \
  --dev-size 300 \
  --num-train-steps 500
```

### Custom Configuration

Train with specific retrieval strategy and parameters:

```bash
python scripts/train_musique_dspy.py train \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --datasets "bdsaglam/musique,answerable,train" \
  --eval-datasets "bdsaglam/musique,answerable,validation" \
  --retriever "semantic" \
  --noise-rate 0.8 \
  --num-hops 3 \
  --train-size 1000 \
  --dev-size 300 \
  --num-train-steps 1000 \
  --learning-rate 2e-5 \
  --kl-beta 0.04 \
  --use-lora
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/train_musique_dspy.py evaluate \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --datasets "bdsaglam/musique,answerable,validation" \
  --test-size 500 \
  --retriever "hybrid" \
  --num-hops 2
```

## Key Differences from Verifiers Approach

1. **DSPy Native**: Uses DSPy's signatures and modules instead of verifiers' tool environment
2. **Multi-hop Reasoning**: Implements multi-step reasoning with query generation and information extraction
3. **Retrieval Integration**: Uses existing MuSiQue retrieve tools but within DSPy's framework
4. **GRPO Training**: Uses DSPy's GRPO implementation with proper MuSiQue metrics

## Parameters

### Model & Training
- `--model`: HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)
- `--port`: Arbor server port (default: 7453)
- `--temperature`: Generation temperature (default: 0.7)

### Dataset
- `--datasets`: Training dataset string format: "name,subset,split" (default: bdsaglam/musique,answerable,train)
- `--eval-datasets`: Optional separate evaluation dataset string (same format)
- `--noise-rate`: Rate for including non-supporting documents (1.0 = all, 0.0 = only supporting)
- `--train-size/--dev-size/--test-size`: Number of examples per split

### Retrieval
- `--retriever`: Strategy: lexical/semantic/hybrid/golden (default: hybrid)
- `--num-hops`: Number of retrieval hops (default: 2)

### Training
- `--num-train-steps`: GRPO training steps (default: 500)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--kl-beta`: KL divergence coefficient (default: 0.04)
- `--use-lora/--no-lora`: Enable LoRA fine-tuning (default: enabled)

## Metrics

The script uses four metrics:
1. **Exact Match**: Fraction of questions answered exactly correctly
2. **F1 Score**: Token-level overlap between predicted and gold answers
3. **Retrieval Recall**: Fraction of supporting documents found
4. **Combined**: Weighted combination prioritizing EM and F1 for answer quality, scaled by question difficulty (number of hops)

The combined metric formula:
- Answer Score = 0.6 × EM + 0.3 × F1  
- Base Score = 0.8 × Answer Score + 0.2 × Retrieval Recall
- Final Score = Base Score × Hop Weight (1x to 2x based on question complexity)

## Output

Results are saved to `./outputs/{run_name}/` including:
- Training logs and checkpoints (managed by Arbor)
- Final evaluation scores
- Example predictions

## Troubleshooting

**Rerank service connection errors**: Ensure `docker-compose up rerank` is running and accessible at `localhost:8931`

**Arbor connection errors**: Verify Arbor server is running at the specified port with the correct model loaded

**Import errors**: Make sure vf_musique is installed: `cd environments/vf_musique && pip install -e .`