import verifiers as vf
from data_processing import prepare_dataset
from metrics import exact_match, extract_all_retrieved_doc_ids, f1
from rerank import RerankClient


# Retrieval tools adapted for modern verifiers
def make_retrieve_tool(name: str = "lexical", top_k: int = 3):
    """Create a retrieve tool function compatible with verifiers ToolEnv."""
    # Initialize rerank client for advanced retrievers
    rerank_client = RerankClient()

    def retrieve_documents(query: str) -> str:
        """
        Retrieve relevant documents by the query. The results get better with more specific queries.

        Args:
            query: The query to retrieve documents for.

        Returns:
            Retrieved documents formatted as text.
        """
        # Get documents from the environment state
        # This will be injected by the environment
        docs = retrieve_documents._docs

        if name == "golden":
            retrieved_docs = [doc for doc in docs if doc["is_supporting"]]
        elif name == "bm25":
            # Use BM25 through rerank service
            texts = [doc["text"] for doc in docs]
            ranking = rerank_client.rerank(query=query, documents=texts, top_n=top_k, model="bm25")
            retrieved_docs = [docs[result.index] for result in ranking.results]
        elif name == "lexical":
            # Use lexical reranking
            texts = [doc["text"] for doc in docs]
            ranking = rerank_client.rerank(query=query, documents=texts, top_n=top_k, model="bm25")
            retrieved_docs = [docs[result.index] for result in ranking.results]
        elif name == "hybrid":
            # Combine semantic and lexical
            texts = [doc["text"] for doc in docs]
            # Get semantic results
            semantic_ranking = rerank_client.rerank(
                query=query, documents=texts, top_n=top_k * 2, model="flashrank/ms-marco-MiniLM-L-12-v2"
            )
            # Get lexical results
            lexical_ranking = rerank_client.rerank(query=query, documents=texts, top_n=top_k * 2, model="bm25")

            # Combine results (simple approach - take best from each)
            semantic_docs = [docs[result.index] for result in semantic_ranking.results[: top_k // 2 + 1]]
            lexical_docs = [docs[result.index] for result in lexical_ranking.results[: top_k // 2 + 1]]

            # Deduplicate and take top_k
            seen_ids = set()
            retrieved_docs = []
            for doc in semantic_docs + lexical_docs:
                if doc["id"] not in seen_ids and len(retrieved_docs) < top_k:
                    retrieved_docs.append(doc)
                    seen_ids.add(doc["id"])
        else:
            raise ValueError(f"Unknown retriever: {name}")

        # Format results
        formatted_docs = []
        for doc in retrieved_docs:
            formatted_docs.append(f"Document ID: {doc['id']}\n{doc['text']}")

        return "\n\n".join(formatted_docs)

    # This will be set by the environment when creating the tool
    retrieve_documents._docs = []
    return retrieve_documents


def make_get_tool():
    """Create a tool to get specific documents by ID."""

    def get_document(doc_id: str) -> str:
        """
        Get a document by its ID.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            The document content.
        """
        docs = get_document._docs
        for doc in docs:
            if doc["id"] == str(doc_id):
                return f"Document ID: {doc['id']}\n{doc['text']}"
        return f"Document with ID {doc_id} not found."

    get_document._docs = []
    return get_document


def make_list_tool():
    """Create a tool to list all available documents."""

    def list_documents() -> str:
        """
        List all available documents (ID and title).

        Returns:
            List of all documents with their IDs and titles.
        """
        docs = list_documents._docs
        doc_list = [f"{doc['id']}. {doc['title']}" for doc in docs]
        return "\n".join(doc_list)

    list_documents._docs = []
    return list_documents


class MuSiQueEnv(vf.ToolEnv):
    """Custom ToolEnv for MuSiQue with document injection."""

    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Set up state and inject documents into tools."""
        state = super().setup_state(state, **kwargs)

        # Get docs from state info
        docs = state.get("info", {}).get("docs", [])
        assert len(docs) > 0, "No docs found in state"

        # Inject documents into tools that need them
        for tool in self.tools:
            if hasattr(tool, "_docs"):
                tool._docs = docs

        return state


# Reward functions for MuSiQue evaluation
def exact_match_reward(completion, answer, info, parser, **kwargs):
    """Exact match reward function."""
    predicted_answer = parser.parse_answer(completion) or ""
    return exact_match(predicted_answer, info["answers"])


def f1_reward(completion, answer, info, parser, **kwargs):
    """F1 score reward function."""
    predicted_answer = parser.parse_answer(completion) or ""
    return f1(predicted_answer, info["answers"])


def weighted_exact_match_reward(completion, answer, info, parser, **kwargs):
    """Weighted exact match reward (harder for multi-hop questions)."""
    predicted_answer = parser.parse_answer(completion) or ""
    em_score = exact_match(predicted_answer, info["answers"])
    n_hops = info.get("n_hops", 1)
    # Weight decreases as number of hops increases (harder questions)
    weight = 1.0 / n_hops if n_hops > 0 else 1.0
    return em_score * weight


def weighted_f1_reward(completion, answer, info, parser, **kwargs):
    """Weighted F1 reward (harder for multi-hop questions)."""
    predicted_answer = parser.parse_answer(completion) or ""
    f1_score = f1(predicted_answer, info["answers"])
    n_hops = info.get("n_hops", 1)
    # Weight decreases as number of hops increases (harder questions)
    weight = 1.0 / n_hops if n_hops > 0 else 1.0
    return f1_score * weight


def retrieval_recall_reward(completion, info, **kwargs):
    """Retrieval recall reward function."""
    supporting_doc_ids = [doc["id"] for doc in info["docs"] if doc["is_supporting"]]
    retrieved_doc_ids = set(extract_all_retrieved_doc_ids(completion))

    if len(retrieved_doc_ids) > 0 and len(supporting_doc_ids) > 0:
        return len(retrieved_doc_ids & set(supporting_doc_ids)) / len(supporting_doc_ids)
    return 0.0


def extract_citations(completion, parser, cite_tag="cite"):
    """Extract citations from the completion using XML parser."""
    assistant_messages = parser.get_assistant_messages(completion)
    if not assistant_messages:
        return []

    # Parse the content to extract citations
    parsed_response = parser.parse(assistant_messages[-1]["content"])
    if hasattr(parsed_response, cite_tag):
        cite_content = getattr(parsed_response, cite_tag)
        if cite_content:
            # Split by comma and clean up
            return [id.strip() for id in cite_content.split(",")]
    return []


def citation_reward(completion, info, parser, **kwargs):
    """Citation reward function based on verifiers citation.py pattern."""
    cited_doc_ids = extract_citations(completion, parser, cite_tag="cite")
    supporting_doc_ids = [doc["id"] for doc in info["docs"] if doc["is_supporting"]]

    if len(supporting_doc_ids) == 0:
        return 1.0  # Perfect score if no citations needed

    if len(cited_doc_ids) == 0:
        return 0.0  # No citations provided

    # Calculate citation recall: how many supporting docs were cited
    cited_supporting = len(set(cited_doc_ids) & set(supporting_doc_ids))
    return cited_supporting / len(supporting_doc_ids)


def combined_reward(*args, **kwargs):
    """Combined reward function."""
    # Get weighted scores, retrieval recall, and citation reward
    weighted_em = weighted_exact_match_reward(*args, **kwargs)
    weighted_f1 = weighted_f1_reward(*args, **kwargs)
    retrieval_recall = retrieval_recall_reward(*args, **kwargs)
    citation_score = citation_reward(*args, **kwargs)

    # Combine: average of weighted EM and F1, plus retrieval and citation bonuses
    return (weighted_em + weighted_f1) / 2 + retrieval_recall * 0.3 + citation_score * 0.2


# Custom rubric for MuSiQue evaluation
def MuSiQueRubric(parser, **kwargs):
    """Create MuSiQue rubric with all reward functions."""
    reward_funcs = [
        exact_match_reward,
        f1_reward,
        retrieval_recall_reward,
        citation_reward,
        combined_reward,
    ]

    # Combined reward gets weight 1, others are for metrics only
    weights = [0.0, 0.0, 0.0, 0.0, 1.0]

    return vf.Rubric(funcs=reward_funcs, weights=weights, parser=parser, **kwargs)


def load_environment(
    datasets_str: str = "bdsaglam/musique,answerable,train",
    eval_datasets_str: str | None = None,
    retriever_name: str = "hybrid",
    retriever_top_k: int = 3,
    noise_rate: float = 1.0,
    n_jobs: int = 1,
    **kwargs,
) -> vf.Environment:
    """Load MuSiQue environment for multi-hop question answering."""

    # Load dataset from datasets_str
    dataset = prepare_dataset(datasets_str, noise_rate=noise_rate)

    # Load evaluation dataset if specified
    eval_dataset = None
    if eval_datasets_str:
        eval_dataset = prepare_dataset(eval_datasets_str, noise_rate=noise_rate)

    # Create tools
    tools = [
        make_retrieve_tool(name=retriever_name, top_k=retriever_top_k),
        make_get_tool(),
        make_list_tool(),
    ]

    # System prompt for MuSiQue
    system_prompt = """Answer the question based on the information provided by tools. You have access to the following tools:
====
{tool_descriptions}
====

For each step:
1. Think through your reasoning inside <think> tags
2. Use tools to retrieve relevant documents
3. Continue until you have found the answer through multi-hop reasoning
4. In the **last** step:
   - Reflect on your previous steps inside <think> tags
   - Cite the documents you used inside <cite> tags by their IDs, e.g. `<cite>1, 2, 3</cite>`
   - Give your final answer inside <answer> tags

- Do not make up tools or arguments that aren't listed.
- Questions require multi-hop reasoning across multiple documents.
- Continue searching until you find all relevant information to answer the question."""

    # Create parser (handles <think>, <cite>, <answer> tags)
    parser = vf.XMLParser(
        fields=["think", "cite", "answer"],
        answer_field="answer",
    )

    # Create environment
    env = MuSiQueEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        tools=tools,
        parser=parser,
        rubric=MuSiQueRubric(parser=parser),
        max_turns=10,
        **kwargs,
    )

    return env
