import random
from typing import List

import verifiers as vf
from data_processing import prepare_datasets
from metrics import exact_match, extract_all_retrieved_doc_ids, f1, get_last_answer
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


class MuSiQueToolEnv(vf.ToolEnv):
    """Custom ToolEnv for MuSiQue with document injection."""

    def __init__(self, tools: List, **kwargs):
        super().__init__(tools=tools, **kwargs)

    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Set up state and inject documents into tools."""
        state = super().setup_state(state, **kwargs)

        # Inject documents into tools that need them
        docs = kwargs.get("docs", [])
        for tool in self.tools:
            if hasattr(tool, "_docs"):
                tool._docs = docs

        return state


# Custom rubrics for MuSiQue evaluation
class MuSiQueRubric(vf.Rubric):
    """Custom rubric for MuSiQue evaluation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score(self, prompt, completion, answer, **kwargs) -> vf.RolloutScore:
        """Score MuSiQue completion with multiple metrics."""
        # Get prediction
        predicted_answer = get_last_answer(completion) or ""

        # Get ground truth answers
        answers = kwargs.get("answers", [answer])
        n_hops = kwargs.get("n_hops", 1)
        docs = kwargs.get("docs", [])

        # Compute EM and F1
        em_score = exact_match(predicted_answer, answers)
        f1_score = f1(predicted_answer, answers)

        # Weight by number of hops (multi-hop questions are harder)
        weighted_em = em_score * min(n_hops / 2, 1)
        weighted_f1 = f1_score * min(n_hops / 2, 1)

        # Compute retrieval recall
        supporting_doc_ids = [doc["id"] for doc in docs if doc["is_supporting"]]
        retrieved_doc_ids = set(extract_all_retrieved_doc_ids(completion))

        retrieval_recall = 0.0
        if len(retrieved_doc_ids) > 0 and len(supporting_doc_ids) > 0:
            retrieval_recall = len(retrieved_doc_ids & set(supporting_doc_ids)) / len(supporting_doc_ids)

        # Combined reward
        reward = (weighted_em + weighted_f1) / 2 + retrieval_recall * 0.5

        return vf.RolloutScore(
            reward=reward,
            metrics={
                "exact_match": em_score,
                "f1": f1_score,
                "weighted_em": weighted_em,
                "weighted_f1": weighted_f1,
                "retrieval_recall": retrieval_recall,
                "n_hops": n_hops,
            },
        )


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
    dataset = prepare_datasets(datasets_str)

    # Apply noise rate filtering (same as in ragent.py)
    dataset = dataset.map(
        lambda x: {"docs": [doc for doc in x["docs"] if doc["is_supporting"] or random.random() < noise_rate]}
    )

    # Load evaluation dataset if specified
    eval_dataset = None
    if eval_datasets_str:
        eval_dataset = prepare_datasets(eval_datasets_str)

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
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
For instance,
<tool>
{{
  "name": "retrieve_documents", 
  "args": {{
    "query": "..."
  }}
}}
</tool>
3. You will see the tool's output inside <result> tags
4. Continue until you have found the answer through multi-hop reasoning
5. In the **last** step:
   - Reflect on your previous steps inside <think> tags
   - Cite the documents you used inside <cite> tags by their IDs, e.g. `<cite>1, 2, 3</cite>`
   - Give your final answer inside <answer> tags

- Tools expect specific JSON input formats.
- Do not make up tools or arguments that aren't listed.
- Questions require multi-hop reasoning across multiple documents.
- Continue searching until you find all relevant information to answer the question."""

    # Create parser (handles <think>, <cite>, <answer> tags)
    parser = vf.XMLParser(
        fields=["think", "cite", "answer"],
        answer_field="answer",
    )

    # Create environment
    env = MuSiQueToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        tools=tools,
        parser=parser,
        max_turns=10,
        rubric=MuSiQueRubric(),
        **kwargs,
    )

    return env
