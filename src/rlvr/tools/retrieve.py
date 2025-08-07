import re
from collections import defaultdict
from typing import Any, Callable, Generator

from verifiers.models import Message, RunContext, Document


def format_doc(doc: Document) -> str:
    return f"Document ID: {doc['id']}\n{doc['text']}"


def golden_retriever(docs: list[Document], query: str) -> list[Document]:
    return [doc for doc in docs if doc["is_supporting"]]


def make_bm25_retriever(
    stemmer_lang: str | None = "en",
    stopwords: list[str] = [],
    top_k: int = 3,
):
    """Create a BM25 retriever function with the given configuration.

    Args:
        stemmer: Language code for stemmer (optional)
        stopwords: List of stopwords to use (optional)
    """
    import bm25s
    import Stemmer

    # Initialize stemmer based on config
    stemmer = None
    if stemmer_lang:
        stemmer = Stemmer.Stemmer(stemmer_lang)

    def retrieve(docs: list[Document], query: str) -> list[Document]:
        """BM25 retriever implementation.

        Args:
            docs: List of documents to search in. Each document should be a dict with 'id' and 'text' fields.
            query: Query string to search for
            top_k: Number of documents to retrieve (default: 3)

        Returns:
            List of documents sorted by relevance score
        """
        k = min(top_k, len(docs))
        retriever = bm25s.BM25(corpus=docs)

        tokenized_corpus = bm25s.tokenize(
            [doc["text"] for doc in docs],
            stopwords=stopwords,
            stemmer=stemmer,
        )
        retriever.index(tokenized_corpus)
        results, scores = retriever.retrieve(bm25s.tokenize(query, stemmer=stemmer), k=k)
        return results[0].tolist()

    return retrieve


def make_rerank_retriever(
    model: str | None = None,
    top_k: int = 3,
    rerank_client=None,
) -> Callable:
    from verifiers.rerank import RerankClient

    rerank_client = rerank_client or RerankClient()

    kwargs: dict[str, Any] = dict(top_n=top_k, return_documents=False)
    if model is not None:
        kwargs["model"] = model

    def retrieve(docs: list[Document], query: str) -> list[Document]:
        texts = [doc["text"] for doc in docs]
        ranking = rerank_client.rerank(query=query, documents=texts, **kwargs)
        return [docs[result.index] for result in ranking.results]

    return retrieve


def combine_retrieval_results(
    ranked_docs_list: list[list[dict]],
    docs: list[dict],
    top_k: int,
) -> list[dict]:
    """Aggregate retrieval results from multiple retrievers.

    Args:
        ranked_docs_list: List of lists containing ranked documents from each retriever.
        docs: List of documents to search in. Each document should be a dict with 'id' and 'text' fields.
        top_k: Number of documents to retrieve.

    Returns:
        List of documents sorted by combined relevance score.
    """
    combined_ranks = defaultdict(int)

    for ranked_docs in ranked_docs_list:
        seen = set()
        for rank, doc in enumerate(ranked_docs):
            idx = doc["id"]
            combined_ranks[idx] += rank
            seen.add(idx)

        # Add top_k + 1 to documents not retrieved by this retriever
        for doc in docs:
            if doc["id"] not in seen:
                combined_ranks[doc["id"]] += top_k

    # Sort results by the summed rank (ascending)
    sorted_results = sorted(combined_ranks.items(), key=lambda item: item[1])
    docs_mapping = {doc["id"]: doc for doc in docs}
    return [docs_mapping[id] for id, _ in sorted_results[:top_k]]


def make_combined_retriever(
    *retrievers: list[Callable],
    top_k: int = 3,
) -> Callable:
    """Create a combined retriever function with the given retrievers and configuration.

    Args:
        retrievers: List of retriever functions to combine.

    Returns:
        A function that retrieves documents by combining results from all retrievers.
    """

    def retrieve(docs: list[Document], query: str) -> list[Document]:
        """Combined retriever implementation.

        Args:
            docs: List of documents to search in. Each document should be a dict with 'id' and 'text' fields.
            query: Query string to search for.
            top_k: Number of documents to retrieve (default: 3).

        Returns:
            List of documents sorted by combined relevance score.
        """
        ranked_docs_list = [retriever(docs, query) for retriever in retrievers]
        return combine_retrieval_results(ranked_docs_list, docs, top_k)

    return retrieve


class RetrieveToolError(Exception):
    pass


def extract_retrieved_doc_ids(content: str) -> list[str]:
    return [id.strip() for id in re.findall(r"^Document ID: (\d+)", content, re.MULTILINE)]


def extract_all_retrieved_doc_ids(trajectory: list[Message]) -> Generator[str, None, None]:
    for msg in trajectory:
        if msg["role"] != "tool":
            continue
        yield from extract_retrieved_doc_ids(msg["content"])


def make_retrieve_tool(name: str = "lexical", top_k: int = 3) -> Callable:
    if name == "golden":
        retriever = golden_retriever
    elif name == "bm25":
        retriever = make_bm25_retriever(top_k=top_k)
    elif name == "hybrid":
        semantic_retriever = make_rerank_retriever(
            model="flashrank/ms-marco-MiniLM-L-12-v2",
            top_k=top_k * 2,
        )
        lexical_retriever = make_rerank_retriever(
            model="bm25",
            top_k=top_k * 2,
        )
        retriever = make_combined_retriever(
            semantic_retriever,
            lexical_retriever,
            top_k=top_k,
        )
    elif name == "hybrid-tei":
        semantic_retriever = make_rerank_retriever(
            model="tei",
            top_k=top_k * 2,
        )
        lexical_retriever = make_rerank_retriever(
            model="bm25",
            top_k=top_k * 2,
        )
        retriever = make_combined_retriever(
            semantic_retriever,
            lexical_retriever,
            top_k=top_k,
        )
    elif name == "lexical":
        retriever = make_rerank_retriever(top_k=top_k, model="bm25")
    elif name.startswith("semantic"):
        kwargs = dict(top_k=top_k)
        parts = name.split("/", 1)
        if len(parts) == 2:
            kwargs["model"] = parts[1]
        retriever = make_rerank_retriever(**kwargs)
    else:
        raise ValueError(f"Invalid retriever name: {name}")

    def retrieve(query: str, run_context: RunContext | None = None, **kwargs) -> str:
        """
        Retrieve for relevant documents by the query. The results get better with more specific queries.
        Args:
            query: The query to retrieve documents for.
        """
        assert run_context is not None, "Run context is required"
        docs = [doc for doc in run_context["input"]["docs"]]
        try:
            retrieved_docs = retriever(docs, query)
        except Exception as e:
            raise RetrieveToolError(f"Error retrieving documents: {e}")
        return "\n\n".join([format_doc(x) for x in retrieved_docs])

    return retrieve


def make_list_tool() -> Callable:
    def list_docs(run_context: RunContext | None = None, **kwargs) -> str:
        """
        List all documents (id and title) available for this question.
        """
        assert run_context is not None, "Run context is required"
        return "\n".join([f"{doc['id']}. {doc['title']}" for doc in run_context["input"]["docs"]])

    return list_docs


def make_get_tool() -> Callable:
    def get_doc(id: str, run_context: RunContext | None = None, **kwargs) -> str:
        """
        Get the document by the id.
        """
        assert run_context is not None, "Run context is required"
        for doc in run_context["input"]["docs"]:
            if doc["id"] == str(id):
                return format_doc(doc)
        raise ValueError(f"Document with id {id} not found")

    return get_doc
