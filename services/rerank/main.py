import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Protocol

import bm25s
import httpx
import Stemmer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rerankers import Reranker
from rerankers.documents import Document
from rerankers.results import RankedResults, Result

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CACHE_DIR = "/tmp/.cache/flashrank"

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "tei")

MODEL_CONFIGS = {
    "bm25": {"warmup": False},
    "flashrank/ms-marco-MiniLM-L-12-v2": {"warmup": True},
    "t5/unicamp-dl/mt5-base-mmarco-v2": {"warmup": True},
    "cross-encoder/mixedbread-ai/mxbai-rerank-base-v1": {"warmup": True},
    "colbert/colbert-ir/colbertv2.0": {"warmup": True},
    "tei": {"warmup": False},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the default model at startup."""
    logger.info("Warming up default model...")
    try:
        reranker = get_reranker(DEFAULT_MODEL)
        logger.info(f"Successfully loaded default model: {DEFAULT_MODEL}")
        if MODEL_CONFIGS[DEFAULT_MODEL]["warmup"]:
            _ = await reranker.rank_async(
                query="What is the capital of France?",
                docs=[
                    "Paris is the capital of France.",
                    "Paris is the capital of France.",
                ],
            )
    except Exception as e:
        logger.error(f"Failed to load default model at startup: {e}")
    yield


app = FastAPI(
    title="Reranker API",
    description="A REST API for document reranking",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all incoming requests and their responses."""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code}")
    return response


class RerankerNotFoundError(Exception):
    pass


class BaseRanker(Protocol):
    async def rank_async(self, query: str, docs: List[str], **kwargs) -> RankedResults: ...


class BM25Ranker(BaseRanker):
    def __init__(
        self,
        stemmer_lang: str | None = "en",
        stopwords: str | list[str] = "english",
        **kwargs,
    ):
        self.stopwords = stopwords
        self.stemmer = Stemmer.Stemmer(stemmer_lang) if stemmer_lang else None

    async def rank_async(self, query: str, docs: List[str], **kwargs):
        """Rank documents using BM25."""
        # Convert docs to format expected by BM25
        doc_dicts = [{"idx": i, "text": doc} for i, doc in enumerate(docs)]

        # Create BM25 instance
        retriever = bm25s.BM25(corpus=doc_dicts)

        # Tokenize corpus and index
        tokenized_corpus = bm25s.tokenize(
            [doc["text"] for doc in doc_dicts],
            stopwords=self.stopwords,
            stemmer=self.stemmer,
        )
        retriever.index(tokenized_corpus)

        # Get results
        documents, scores = retriever.retrieve(
            bm25s.tokenize(query, stemmer=self.stemmer),
            k=len(docs),
        )

        ranked_docs = [
            Result(
                document=Document(doc_id=doc["idx"], text=doc["text"]),
                score=score,
            )
            for doc, score in zip(documents[0].tolist(), scores[0].tolist())
        ]
        return RankedResults(results=ranked_docs, query=query, has_scores=True)


class TEIRanker(BaseRanker):
    """Handles reranking using a TEI endpoint."""

    def __init__(self, tei_url: str | None = None):
        if tei_url is None:
            tei_url = os.getenv("TEI_RERANK_URL")
        if tei_url is None:
            raise ValueError("TEI URL must be provided or set in the TEI_RERANK_URL environment variable")
        self.tei_url = tei_url.rstrip("/")

    async def rank_async(self, query: str, docs: List[str], **kwargs):
        """Rank documents using the TEI /rerank endpoint."""
        url = f"{self.tei_url}/rerank"

        payload = {"query": query, "texts": docs, "raw_scores": False}
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=30.0)) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()  # Raise exception for non-200 status codes
            except httpx.RequestError as e:
                logger.error(f"TEI request failed: {e}")
                raise RuntimeError(f"Error communicating with TEI endpoint: {e}")
            except httpx.HTTPStatusError as e:
                logger.error(f"TEI request failed with status {e.response.status_code}: {e.response.text}")
                raise RuntimeError(f"TEI endpoint returned error: {e.response.status_code} - {e.response.text}")

        # Parse TEI list response (assuming TeiRerankItem structure like in verifier)
        try:
            # Minimal Pydantic-like validation for the expected structure
            tei_response_data = response.json()
            if not isinstance(tei_response_data, list):
                raise ValueError("TEI response is not a list")
            tei_results = []
            for item in tei_response_data:
                if not isinstance(item, dict) or "index" not in item or "score" not in item:
                    raise ValueError("Invalid item structure in TEI response")
                tei_results.append(item)  # Keep as dict for simplicity here

        except Exception as e:
            logger.error(f"Failed to parse TEI response: {e}")
            raise RuntimeError(f"Invalid response format from TEI endpoint: {e}")

        # Adapt TEI response back to rerankers.results structure
        ranked_docs = [
            Result(
                document=Document(doc_id=item["index"], text=docs[item["index"]]),
                score=item["score"],
            )
            for item in tei_results
        ]

        # TEI already returns sorted results by score, descending
        return RankedResults(results=ranked_docs, query=query, has_scores=True)


# Initialize reranker models
rerankers: dict[str, BaseRanker] = {}


def get_reranker(model_id: str) -> BaseRanker:
    """Get or initialize a reranker model."""
    if model_id == "bm25":
        if model_id not in rerankers:
            logger.info("Loading BM25 reranker")
            try:
                rerankers[model_id] = BM25Ranker()
                if rerankers[model_id] is None:
                    raise RerankerNotFoundError("BM25 Reranker could not be initialized")
            except Exception as e:
                logger.error(f"Failed to load BM25 reranker: {e}")
                raise RuntimeError(f"BM25 reranker could not be loaded: {str(e)}")
        return rerankers[model_id]

    # Check for TEI models
    if model_id.startswith("tei"):
        if model_id not in rerankers:
            logger.info(f"Initializing TEI reranker proxy for model: {model_id}")
            try:
                # We use the TEI URL, the specific model name after 'tei' isn't used by the TeiRanker itself
                # but keeping it in the key `rerankers[model_id]` allows tracking specific TEI models if needed later.
                rerankers[model_id] = TEIRanker()
            except Exception as e:
                logger.error(f"Failed to initialize TEI reranker proxy: {e}")
                raise RuntimeError(f"TEI reranker proxy could not be initialized: {str(e)}")
        return rerankers[model_id]

    model_type, model_name = model_id.split("/", 1)
    if model_id not in rerankers:
        logger.info(f"Loading reranker model: {model_id}")
        try:
            rerankers[model_id] = Reranker(
                model_name,
                model_type=model_type,
                cache_dir=CACHE_DIR,
            )
            if rerankers[model_id] is None:
                raise RerankerNotFoundError("Reranker could not be initialized")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Model '{model_id}' could not be loaded: {str(e)}")
    return rerankers[model_id]


class RerankRequest(BaseModel):
    model: str = DEFAULT_MODEL
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str]


class RerankResponse(BaseModel):
    results: List[RerankResult]
    meta: Dict[str, Any] = {}


@app.post("/rerank")
async def rerank(request: RerankRequest) -> RerankResponse:
    """Rerank documents based on their relevance to a query."""
    try:
        reranker = get_reranker(request.model)

        # Rerank the documents
        ranking = await reranker.rank_async(
            query=request.query,
            docs=request.documents,
        )

        # Format the response according to Cohere's schema
        rerank_results = []
        for result in ranking.results[: request.top_n]:
            score = result.score if ranking.has_scores else 1
            response_result = RerankResult(
                index=result.doc_id,
                relevance_score=score,
                document=None if not request.return_documents else request.documents[result.doc_id],
            )
            rerank_results.append(response_result)

        return RerankResponse(
            results=rerank_results,
            meta={"model": request.model},
        )
    except RerankerNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List available models endpoint."""
    try:
        return {
            "models": sorted(list(MODEL_CONFIGS.keys())),
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
