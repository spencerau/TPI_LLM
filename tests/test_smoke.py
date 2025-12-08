import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_rag.retrieval import UnifiedRAG
from core_rag.utils import load_config


@pytest.fixture(scope="module")
def rag():
    return UnifiedRAG()


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestSmoke:
    
    def test_rag_initializes(self, rag):
        assert rag is not None
        assert rag.config is not None
        assert rag.client is not None
        assert rag.ollama_api is not None

    def test_collections_accessible(self, rag, config):
        collections = config.get('qdrant', {}).get('collections', {})
        
        for name, collection_name in collections.items():
            try:
                info = rag.client.get_collection(collection_name)
                print(f"{name}: {info.points_count} documents")
            except Exception as e:
                pytest.skip(f"Collection {collection_name} not available: {e}")

    def test_search_returns_results(self, rag, config):
        collections = list(config.get('qdrant', {}).get('collections', {}).values())
        if not collections:
            pytest.skip("No collections configured")
        
        collection_name = collections[0]
        
        try:
            info = rag.client.get_collection(collection_name)
            if info.points_count == 0:
                pytest.skip(f"Collection {collection_name} is empty")
        except Exception:
            pytest.skip(f"Collection {collection_name} not available")
        
        results = rag.search_collection(
            query="job interview accommodations",
            collection_name=collection_name,
            top_k=3
        )
        
        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0
        
        for i, r in enumerate(results, 1):
            score = r.get('score', 0)
            text = r.get('payload', {}).get('chunk_text', '')[:100]
            print(f"  {i}. Score: {score:.4f} - {text}...")

    def test_answer_generation(self, rag, config):
        collections = list(config.get('qdrant', {}).get('collections', {}).values())
        if not collections:
            pytest.skip("No collections configured")
        
        try:
            info = rag.client.get_collection(collections[0])
            if info.points_count == 0:
                pytest.skip("No documents ingested yet")
        except Exception:
            pytest.skip("Collections not available")
        
        query = "What are some tips for job interviews for people with disabilities?"
        result = rag.answer_question(query, stream=False)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "I couldn't generate a response."
        
        print(f"Query: {query}")
        print(f"Response: {result[:500]}...")
