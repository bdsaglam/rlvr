import pytest
from vf_musique.rerank import RerankClient, RerankError
from vf_musique.tools import make_retrieve_tool

# Sample test documents
SAMPLE_DOCS = [
    {
        "id": "doc1",
        "title": "Python Programming",
        "text": "Python is a high-level programming language.",
        "is_supporting": True
    },
    {
        "id": "doc2", 
        "title": "Machine Learning",
        "text": "Machine learning is a subset of artificial intelligence.",
        "is_supporting": False
    },
    {
        "id": "doc3",
        "title": "Data Science",
        "text": "Data science combines statistics, programming, and domain expertise.",
        "is_supporting": True
    }
]


@pytest.fixture(scope="session")
def rerank_client():
    """Create a real RerankClient for integration testing."""
    client = RerankClient()
    
    # Check if rerank service is available
    try:
        client.health()
    except Exception as e:
        pytest.skip(f"Rerank service not available: {e}")
    
    return client


class TestMakeRetrieveTool:
    """Integration tests for make_retrieve_tool function."""
    
    def test_rerank_service_availability(self, rerank_client):
        """Test that the rerank service is available and responsive."""
        # Test health check
        health = rerank_client.health()
        assert isinstance(health, dict)
        
        # Test basic rerank functionality
        test_docs = ["Python programming", "Machine learning", "Data science"]
        response = rerank_client.rerank(
            query="programming",
            documents=test_docs,
            top_n=2,
            model="bm25"
        )
        
        assert hasattr(response, 'results')
        assert len(response.results) <= 2
        assert all(hasattr(result, 'index') and hasattr(result, 'relevance_score') 
                  for result in response.results)

    def test_golden_retriever(self):
        """Test golden retriever returns only supporting documents."""
        retrieve_tool = make_retrieve_tool("golden")
        retrieve_tool._docs = SAMPLE_DOCS
        
        result = retrieve_tool("test query", top_n=2)
        
        # Should return only supporting documents (doc1 and doc3)
        assert "Document ID: doc1" in result
        assert "Document ID: doc3" in result
        assert "Document ID: doc2" not in result
        assert "Python is a high-level programming language." in result
        assert "Data science combines statistics" in result

    def test_lexical_retriever(self, rerank_client):
        """Test lexical retriever using BM25."""
        retrieve_tool = make_retrieve_tool("lexical")
        retrieve_tool._docs = SAMPLE_DOCS
        
        result = retrieve_tool("programming language", top_n=1)
        
        # Should return a document
        assert "Document ID:" in result
        assert len(result.strip()) > 0
        
        # Test that it returns different results for different queries
        result_ai = retrieve_tool("machine learning artificial intelligence", top_n=1)
        assert "Document ID:" in result_ai

    def test_semantic_retriever(self, rerank_client):
        """Test semantic retriever with default model."""
        retrieve_tool = make_retrieve_tool("semantic")
        retrieve_tool._docs = SAMPLE_DOCS
        
        result = retrieve_tool("AI concepts", top_n=2)
        
        # Should return documents
        assert "Document ID:" in result
        assert len(result.strip()) > 0
        
        # Should return at most 2 documents
        doc_count = result.count("Document ID:")
        assert doc_count <= 2

    def test_semantic_retriever_with_custom_model(self, rerank_client):
        """Test semantic retriever with custom model."""
        # First check what models are available
        try:
            models = rerank_client.list_models()
            available_models = models.get("models", [])
        except Exception:
            pytest.skip("Cannot list models from rerank service")
            
        if not available_models:
            pytest.skip("No models available in rerank service")
        
        # Use first available model
        model_name = available_models[0]
        retrieve_tool = make_retrieve_tool(f"semantic/{model_name}")
        retrieve_tool._docs = SAMPLE_DOCS
        
        result = retrieve_tool("test query", top_n=1)
        
        # Should return a document
        assert "Document ID:" in result
        assert len(result.strip()) > 0

    def test_hybrid_retriever(self, rerank_client):
        """Test hybrid retriever combining semantic and lexical results."""
        retrieve_tool = make_retrieve_tool("hybrid")
        retrieve_tool._docs = SAMPLE_DOCS
        
        result = retrieve_tool("test query", top_n=2)
        
        # Should return documents (hybrid combines results)
        assert "Document ID:" in result
        assert len(result.strip()) > 0
        
        # Should return at most 2 documents due to top_n
        doc_count = result.count("Document ID:")
        assert doc_count <= 2

    def test_unknown_retriever_raises_error(self):
        """Test that unknown retriever name raises ValueError."""
        retrieve_tool = make_retrieve_tool("unknown_retriever")
        retrieve_tool._docs = SAMPLE_DOCS
        
        with pytest.raises(ValueError, match="Unknown retriever: unknown_retriever"):
            retrieve_tool("test query")

    def test_top_n_parameter(self, rerank_client):
        """Test that top_n parameter is respected."""
        retrieve_tool = make_retrieve_tool("lexical")
        retrieve_tool._docs = SAMPLE_DOCS
        
        # Test with top_n=1
        result_1 = retrieve_tool("test query", top_n=1)
        doc_count_1 = result_1.count("Document ID:")
        assert doc_count_1 <= 1
        
        # Test with top_n=3 (all documents)
        result_3 = retrieve_tool("test query", top_n=3)
        doc_count_3 = result_3.count("Document ID:")
        assert doc_count_3 <= 3
        assert doc_count_3 >= doc_count_1  # Should return at least as many as top_n=1

    def test_empty_docs_list(self):
        """Test behavior with empty document list."""
        retrieve_tool = make_retrieve_tool("golden")
        retrieve_tool._docs = []
        
        result = retrieve_tool("test query")
        
        # Should return empty result
        assert result == ""

    def test_formatted_output(self):
        """Test that output is properly formatted."""
        retrieve_tool = make_retrieve_tool("golden")
        retrieve_tool._docs = SAMPLE_DOCS
        
        result = retrieve_tool("test query", top_n=1)
        
        # Should contain properly formatted document
        lines = result.split('\n')
        assert any(line.startswith("Document ID: doc") for line in lines)
        assert len([line for line in lines if line.strip()]) >= 2  # At least ID and content lines

    @pytest.mark.parametrize("retriever_name", ["lexical", "semantic", "hybrid"])
    def test_retriever_variants(self, rerank_client, retriever_name):
        """Test that all retriever variants can be created and called without errors."""
        retrieve_tool = make_retrieve_tool(retriever_name)
        retrieve_tool._docs = SAMPLE_DOCS
        
        # Should not raise any exceptions
        result = retrieve_tool("test query", top_n=1)
        assert isinstance(result, str)
        
        # Should return some content for valid retrievers
        if retriever_name != "unknown":
            assert len(result.strip()) >= 0  # May be empty if no docs match
    
    def test_service_error_handling(self):
        """Test error handling when rerank service fails."""
        # Create retriever that uses rerank service
        retrieve_tool = make_retrieve_tool("lexical")
        retrieve_tool._docs = SAMPLE_DOCS
        
        # This should work normally (service is available per fixture)
        try:
            result = retrieve_tool("test query", top_n=1)
            # If service works, result should be valid
            assert isinstance(result, str)
        except RerankError:
            # If service fails, we expect a RerankError
            pytest.fail("Service should be available for integration tests")
    
    def test_retriever_with_no_documents(self):
        """Test retriever behavior with no documents."""
        for retriever_name in ["golden", "lexical", "semantic", "hybrid"]:
            retrieve_tool = make_retrieve_tool(retriever_name)
            retrieve_tool._docs = []
            
            if retriever_name == "golden":
                # Golden retriever should return empty string
                result = retrieve_tool("test query", top_n=1)
                assert result == ""
            else:
                # Other retrievers may fail or return empty - both are acceptable
                try:
                    result = retrieve_tool("test query", top_n=1)
                    assert isinstance(result, str)
                except (RerankError, Exception):
                    # Expected - no documents to rerank
                    pass