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

### Prerequisites

```bash
# Install the verifiers library
pip install verifiers[all]

# Install additional dependencies
pip install datasets requests
```

### Environment Installation

```bash
# Install the MuSiQue environment package
vf-install environments/vf_musique
```

## Usage

### Quick Evaluation

```bash
# Evaluate with default model
vf-eval vf-musique

# Evaluate with specific model
vf-eval vf-musique --model gpt-4o-mini
```

### Training

The training script uses Typer for a modern CLI experience with multiple commands:

**Basic training:**
```bash
# Default command is 'train'
python scripts/train_musique.py --model meta-llama/Llama-3.1-8B-Instruct

# Explicit train command
python scripts/train_musique.py train --model meta-llama/Llama-3.1-8B-Instruct
```

**Advanced training with custom settings:**
```bash
python scripts/train_musique.py train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --retriever hybrid \
    --num-train-examples 2000 \
    --batch-size 16 \
    --learning-rate 1e-6 \
    --num-epochs 3 \
    --use-lora \
    --lora-r 32 \
    --lora-alpha 64 \
    --push-to-hub
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

## File Structure

```
rlvr/
├── environments/vf_musique/          # Installable MuSiQue environment
│   ├── __init__.py
│   ├── vf_musique.py                # Main environment implementation
│   ├── metrics.py                   # Evaluation metrics
│   ├── pyproject.toml              # Package configuration
│   └── README.md                   # Environment documentation
├── scripts/
│   ├── train_musique.py            # Training script
│   └── evaluate_musique.py         # Evaluation script
├── src/rlvr/                       # Legacy components (reused selectively)
│   ├── clients/                    # API clients (rerank, wiki search)
│   ├── datasets/                   # Dataset preprocessing
│   ├── tools/                      # Tool implementations
│   └── rubrics/                    # Reward functions
└── CLAUDE.md                       # This documentation
```

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
# Test environment loading
python test_simple_structure.py

# Debug with minimal examples
python scripts/train_musique.py --num-train-examples 10 --max-steps 5
```

## Future Enhancements

### Planned Features
- [ ] Additional retrieval strategies (dense passage retrieval)
- [ ] Multi-dataset support (HotpotQA, 2WikiMultiHopQA)
- [ ] Advanced reward shaping techniques
- [ ] Integration with external knowledge bases
- [ ] Distributed training support

### Research Directions
- [ ] Self-supervised document filtering
- [ ] Hierarchical reasoning decomposition  
- [ ] Meta-learning for few-shot adaptation
- [ ] Interpretability and reasoning visualization

## Contributing

This project builds on the excellent [verifiers library](https://github.com/willccbb/verifiers). When contributing:

1. Follow verifiers conventions for environment design
2. Maintain compatibility with standard rubric protocols
3. Test with multiple retrieval strategies
4. Document new features in environment README

## References

- [Verifiers Library](https://github.com/willccbb/verifiers) - Official documentation
- [MuSiQue Dataset](https://github.com/StonyBrookNLP/musique) - Multi-hop QA dataset
- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Reinforcement learning algorithm