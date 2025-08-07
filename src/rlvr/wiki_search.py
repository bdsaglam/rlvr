import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel


class WikiSearchError(Exception):
    pass


class SearchDocument(BaseModel):
    id: str
    title: str
    body: str


class SearchResult(BaseModel):
    rank: int
    document: SearchDocument
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    meta: Dict[str, Any] = {}


class WikiSearchClient:
    """Client for the Wiki Search API service."""

    def __init__(self, base_url: str | None = None, api_key: Optional[str] = None):
        """
        Initialize the wiki search client.

        Args:
            base_url: The base URL for the wiki search API.
                      Defaults to WIKI_SEARCH_API_URL environment variable or "http://localhost:8932"
            api_key: Optional API key for auth. Defaults to WIKI_SEARCH_API_KEY environment variable.
        """
        self.base_url = base_url or os.getenv("WIKI_SEARCH_API_URL") or "http://localhost:8932"
        self.api_key = api_key or os.getenv("WIKI_SEARCH_API_KEY")

        # Strip trailing slash if present
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def search(
        self,
        query: str,
        top_n: Optional[int] = None,
    ) -> SearchResponse:
        """
        Search for documents based on a query.

        Args:
            query: The search query.
            top_n: Number of top results to return. If None, returns all results.

        Returns:
            Dict containing search results and metadata.
        """
        url = f"{self.base_url}/search"

        payload: dict[str, Any] = {"query": query}

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

            raise WikiSearchError(f"API request failed with status {response.status_code}: {error_detail}")

        return SearchResponse.model_validate(response.json())

    def health(self) -> Dict[str, str]:
        """Check the health status of the wiki search API."""
        url = f"{self.base_url}/health"

        with httpx.Client() as client:
            response = client.get(url)

        if response.status_code != 200:
            raise RuntimeError(f"Health check failed with status {response.status_code}")

        return response.json()
