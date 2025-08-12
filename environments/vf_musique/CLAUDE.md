## MuSiQue Environment

- Custom `ToolEnv` implementation for multi-hop question answering
- Document retrieval tools (BM25, semantic, hybrid retrieval)
- MuSiQue-specific dataset preprocessing
- Citation tracking and multi-hop reasoning support

## Key Features

### Multi-Step Reasoning
- **Tool-Based Interaction**: Models use retrieval tools to gather information
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