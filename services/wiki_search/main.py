import json
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher

# Set Java options
os.environ["JAVA_OPTS"] = "-Xms16g -Xmx32g -XX:MaxDirectMemorySize=16g"

app = FastAPI(title="Wiki Search API")

# Initialize searcher
try:
    _searcher = LuceneSearcher.from_prebuilt_index("wikipedia-kilt-doc")
except Exception as e:
    print(f"Error initializing searcher: {e}")
    _searcher = None


class SearchRequest(BaseModel):
    query: str
    top_n: Optional[int] = 3


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
    metadata: dict = {}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    return {"status": "healthy"}


@app.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search Wikipedia articles.

    Args:
        request: Search request containing query and optional top_n
    """
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        hits = _searcher.search(request.query, k=request.top_n)
        if not hits:
            return SearchResponse(results=[])

        results = []
        for i, hit in enumerate(hits):
            doc = _searcher.doc(hit.docid)
            contents = json.loads(doc.raw())
            title = contents.get("title", "")
            body = contents.get("contents", "")
            if not title:
                title, body = body.split("\n", 1)
                title = title.strip()
                body = body.strip()

            results.append(
                SearchResult(
                    rank=i + 1,
                    score=hit.score,
                    document=SearchDocument(
                        id=hit.docid,
                        title=title,
                        body=body,
                    ),
                )
            )

        return SearchResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
