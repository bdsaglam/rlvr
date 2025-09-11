# VF MuSiQue

MuSiQue environment for multi-hop question answering with the verifiers library.

## Features

- **Multi-Step Reasoning**: Models use retrieval tools to gather information
- **Multi-Hop Questions**: Dataset requires connecting information across multiple documents
- **Tool-Based Interaction**: Document retrieval tools (BM25, semantic, hybrid retrieval)
- **Citation Tracking**: Environment tracks and rewards proper citations
- **Completion Detection**: Environment detects when reasoning is complete

## Installation

```bash
uv add --editable ./environments/vf_musique
```

## Usage

```python
import verifiers as vf

# Load the environment
env = vf.load_environment("vf-musique")

# Use with GRPO training or evaluation
```

## Retrieval Strategies

- **BM25**: Classic lexical retrieval
- **Semantic**: Neural embedding-based retrieval  
- **Hybrid**: Combination of lexical and semantic approaches
- **Golden**: Oracle retrieval for debugging (returns supporting documents)

## Evaluation Metrics

- **Exact Match**: Binary accuracy against ground truth
- **F1 Score**: Token-level overlap with references
- **Retrieval Recall**: Fraction of supporting documents retrieved
- **Citation Reward**: Reward for proper citations
- **Weighted Scoring**: Multi-hop questions receive higher weight
- **Combined Reward**: Balances answer quality and retrieval performance