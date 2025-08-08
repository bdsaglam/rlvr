# RLVR: Reinforcement Learning with Verifier Rewards

## Project Overview

RLVR is a fresh implementation of multi-step question answering training using the modern [verifiers library](https://github.com/willccbb/verifiers). This project focuses on training language models with reinforcement learning on the MuSiQue dataset for multi-hop reasoning tasks.

### Background

This project represents a migration from a custom verifiers fork to the official verifiers library, taking advantage of modern improvements like:
- Native tool calling support
- Clean environment protocols (`ToolEnv`, `MultiTurnEnv`)
- Modern GRPO trainer with async batch generation
- Installable environment packages
- Proper rubric system for evaluation

## Architecture

### Core Components

1. **MuSiQue Environment** (`environments/vf_musique/`)
   - Custom `ToolEnv` implementation for multi-hop question answering
   - Document retrieval tools (BM25, semantic, hybrid retrieval)
   - MuSiQue-specific dataset preprocessing
   - Citation tracking and multi-hop reasoning support

2. **Training Infrastructure** (`scripts/train_musique.py`)
   - Modern GRPO training using official verifiers library
   - LoRA fine-tuning support
   - Configurable retrieval strategies
   - WandB integration for experiment tracking

3. **Evaluation System**
   - Custom rubrics for MuSiQue evaluation
   - Exact match and F1 scoring
   - Retrieval quality metrics (recall, precision)
   - Multi-hop difficulty weighting

## Installation & Setup

### Setup Python Environment

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install flash-attn --no-build-isolation
```

### Environment Installation

```bash
# Install the MuSiQue environment package
vf-install vf-musique -p environments
```

## Usage

### Quick Evaluation

```bash
vf-eval vf-musique --model meta-llama/Llama-3.1-8B-Instruct
```

### Training

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

### Evaluation of Trained Models

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

**Benchmark multiple models:**
```bash
python scripts/evaluate_musique.py benchmark \
    --models model1,model2,model3 \
    --retrievers bm25,hybrid \
    --output-dir benchmark_results
```

**Analyze results:**
```bash
python scripts/evaluate_musique.py analyze benchmark_results
```

## Key Features

### Multi-Step Reasoning
- **Tool-Based Interaction**: Models use retrieval tools to gather information
- **Citation Requirements**: Models must cite sources used in reasoning
- **Multi-Hop Questions**: Dataset requires connecting information across multiple documents
- **Completion Detection**: Environment detects when reasoning is complete

### Retrieval Strategies
- **BM25**: Classic lexical retrieval
- **Semantic**: Neural embedding-based retrieval
- **Hybrid**: Combination of lexical and semantic approaches
- **Golden**: Oracle retrieval for debugging (returns supporting documents)

### Evaluation Metrics
- **Exact Match**: Binary accuracy against ground truth
- **F1 Score**: Token-level overlap with references
- **Retrieval Recall**: Fraction of supporting documents retrieved
- **Weighted Scoring**: Multi-hop questions receive higher weight
- **Combined Reward**: Balances answer quality and retrieval performance

## Technical Details

### Environment Implementation

The MuSiQue environment (`MuSiQueToolEnv`) extends verifiers' `ToolEnv` to provide:

1. **Document Injection**: Tools receive access to question-specific documents
2. **Multi-Turn Interaction**: Up to 10 turns of tool usage per question
3. **Native Tool Calling**: Python functions automatically converted to OpenAI format
4. **State Management**: Tracks retrieval history and completion status

### Tool Integration

Tools are implemented as Python functions with docstrings that become tool descriptions:

```python
def retrieve_documents(query: str) -> str:
    """
    Retrieve relevant documents by the query.
    
    Args:
        query: The query to retrieve documents for.
    
    Returns:
        Retrieved documents formatted as text.
    """
    # Implementation...
```

### Custom Rubrics

The `MuSiQueRubric` class provides comprehensive evaluation:

```python
def score(self, prompt, completion, answer, **kwargs) -> vf.RolloutScore:
    # Compute EM, F1, retrieval metrics
    # Weight by question difficulty (number of hops)
    # Return combined reward and detailed metrics
```

## Migration Benefits

### From Custom Fork to Official Library

**Before (Custom Fork)**:
- Complex XML parsing and tool integration
- Manual environment state management
- Custom training loops and reward computation
- Difficulty staying up-to-date with improvements

**After (Official Verifiers)**:
- Native tool calling with automatic OpenAI conversion
- Clean environment protocols and state management
- Modern GRPO trainer with async batch generation
- Easy updates and community improvements

### Performance Improvements

- **Async Training**: Improved throughput with async batch generation
- **Native Tools**: Cleaner tool integration without custom parsers
- **Modern Config**: Better hyperparameter management
- **Standardized Evaluation**: Consistent metrics across environments

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure verifiers library is installed and environment is in Python path
2. **Service Dependencies**: Rerank and wiki search services must be running for advanced retrievers
3. **Dataset Loading**: MuSiQue dataset download may take time on first run
4. **GPU Memory**: Adjust batch size and gradient accumulation for available hardware

### Debug Mode

```bash
# Debug with minimal examples
python scripts/train_musique.py --num-train-examples 10 --max-steps 5
```

## Future Enhancements

### Planned Features
- [ ] Multi-dataset support (HotpotQA, 2WikiMultiHopQA)
- [ ] Advanced reward shaping techniques
- [ ] Integration with external knowledge bases

### Research Directions
- [ ] Self-supervised document filtering
- [ ] Hierarchical reasoning decomposition  
- [ ] Meta-learning for few-shot adaptation
- [ ] Interpretability and reasoning visualization

## References

- [Verifiers Library](https://github.com/willccbb/verifiers) - Official documentation
- [MuSiQue Dataset](https://github.com/StonyBrookNLP/musique) - Multi-hop QA dataset
- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Reinforcement learning algorithm