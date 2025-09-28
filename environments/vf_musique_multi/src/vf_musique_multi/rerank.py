import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel


class RerankError(Exception):
    pass


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str]


class RerankResponse(BaseModel):
    results: List[RerankResult]
    meta: Dict[str, Any] = {}


class RerankClient:
    """Client for the Reranker API service."""

    def __init__(self, base_url: str = None, api_key: Optional[str] = None):
        """
        Initialize the rerank client.

        Args:
            base_url: The base URL for the reranker API.
                      Defaults to RERANK_API_URL environment variable or "http://localhost:8931"
            api_key: Optional API key for auth. Defaults to RERANK_API_KEY environment variable.
        """
        self.base_url = base_url or os.getenv("RERANK_API_URL", "http://localhost:8931")
        self.api_key = api_key or os.getenv("RERANK_API_KEY")

        # Strip trailing slash if present
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str | None = None,
        top_n: Optional[int] = None,
        return_documents: bool = True,
    ) -> RerankResponse:
        """
        Rerank documents based on their relevance to a query.

        Args:
            query: The query to use for reranking.
            documents: The list of documents to rerank.
            model: The reranker model to use.
            top_n: Number of top results to return. If None, returns all results.
            return_documents: Whether to include the document text in the results.

        Returns:
            Dict containing reranked results and metadata.
        """
        url = f"{self.base_url}/rerank"

        payload = {"query": query, "documents": documents, "return_documents": return_documents}
        if model is not None:
            payload["model"] = model

        if top_n is not None:
            payload["top_n"] = top_n

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=httpx.Timeout(timeout=30.0)) as client:
            response = client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text

            raise RerankError(f"API request failed with status {response.status_code}: {error_detail}")

        return RerankResponse.model_validate(response.json())

    def health(self) -> Dict[str, str]:
        """Check the health status of the reranker API."""
        url = f"{self.base_url}/health"

        with httpx.Client() as client:
            response = client.get(url)

        if response.status_code != 200:
            raise RuntimeError(f"Health check failed with status {response.status_code}")

        return response.json()

    def list_models(self) -> Dict[str, List[str]]:
        """Get the list of available models."""
        url = f"{self.base_url}/models"

        with httpx.Client() as client:
            response = client.get(url)

        if response.status_code != 200:
            raise Exception(f"List models request failed with status {response.status_code}")

        return response.json()
