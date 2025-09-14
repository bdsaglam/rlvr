from typing import TypedDict

from agents import RunContextWrapper

from .data import MuSiQueInfo
from .rerank import RerankClient


class ToolContext(TypedDict):
    info: MuSiQueInfo


def make_retrieve_tool(name: str = "lexical", default_top_n: int = 1):
    """Create a retrieve tool function compatible with verifiers ToolEnv."""
    # Initialize rerank client for advanced retrievers
    rerank_client = RerankClient()

    def retrieve_documents(ctx: RunContextWrapper[ToolContext], query: str) -> str:
        """
        Retrieve documents by the query. The results get better with more specific queries.

        Args:
            query: The query to retrieve documents for.

        Returns:
            Retrieved documents formatted as text.
        """
        top_n = default_top_n

        # Get documents from the injected state
        docs = ctx.get("info", {}).get("docs", [])

        if name == "golden":
            retrieved_docs = [doc for doc in docs if doc["is_supporting"]]
        elif name == "lexical":
            # Use lexical reranking
            texts = [doc["text"] for doc in docs]
            ranking = rerank_client.rerank(
                query=query,
                documents=texts,
                top_n=top_n,
                model="bm25",
            )
            retrieved_docs = [docs[result.index] for result in ranking.results]
        elif name.startswith("semantic"):
            kwargs = dict()
            parts = name.split("/", 1)
            if len(parts) == 2:
                kwargs["model"] = parts[1]
            texts = [doc["text"] for doc in docs]
            ranking = rerank_client.rerank(
                query=query,
                documents=texts,
                top_n=top_n,
                **kwargs,
            )
            retrieved_docs = [docs[result.index] for result in ranking.results]
        elif name == "hybrid":
            # Combine semantic and lexical
            texts = [doc["text"] for doc in docs]
            # Get semantic results
            semantic_ranking = rerank_client.rerank(query=query, documents=texts, top_n=top_n * 2, model="tei")
            # Get lexical results
            lexical_ranking = rerank_client.rerank(query=query, documents=texts, top_n=top_n * 2, model="bm25")

            # Combine results (simple approach - take best from each)
            semantic_docs = [docs[result.index] for result in semantic_ranking.results[: top_n // 2 + 1]]
            lexical_docs = [docs[result.index] for result in lexical_ranking.results[: top_n // 2 + 1]]

            # Deduplicate and take top_n
            seen_ids = set()
            retrieved_docs = []
            for doc in semantic_docs + lexical_docs:
                if doc["id"] not in seen_ids and len(retrieved_docs) < top_n:
                    retrieved_docs.append(doc)
                    seen_ids.add(doc["id"])
        else:
            raise ValueError(f"Unknown retriever: {name}")

        # Format results
        formatted_docs = []
        for doc in retrieved_docs:
            formatted_docs.append(f"Document ID: {doc['id']}\n{doc['text']}")

        return "\n\n".join(formatted_docs)

    return retrieve_documents


def make_get_tool():
    """Create a tool to get specific documents by ID."""

    def get_document(ctx: RunContextWrapper[ToolContext], doc_id: str) -> str:
        """
        Get a document by its ID.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            The document content.
        """
        # Get documents from the injected state
        docs = ctx.get("info", {}).get("docs", [])
        for doc in docs:
            if doc["id"] == str(doc_id):
                return f"Document ID: {doc['id']}\n{doc['text']}"
        return f"Document with ID {doc_id} not found."

    return get_document


def complete(reasoning: str, cited_doc_ids: list[str], final_answer: str) -> str: ...
