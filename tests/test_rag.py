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


def test_rag_initialization(rag):
    assert rag is not None
    assert rag.config is not None
    assert rag.client is not None
    assert rag.ollama_api is not None


def test_config_loads(config):
    assert config is not None
    assert 'qdrant' in config
    assert 'embedding' in config
    assert 'llm' in config


def test_collections_exist(rag, config):
    collections = config.get('qdrant', {}).get('collections', {})
    assert len(collections) > 0
    
    for name, collection_name in collections.items():
        try:
            info = rag.client.get_collection(collection_name)
            assert info is not None
        except Exception:
            pytest.skip(f"Collection {collection_name} not yet created")


def test_search_collection(rag, config):
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
        query="job interview",
        collection_name=collection_name,
        top_k=3
    )
    
    assert results is not None
    assert isinstance(results, list)


def test_answer_question(rag, config):
    collections = list(config.get('qdrant', {}).get('collections', {}).values())
    if not collections:
        pytest.skip("No collections configured")
    
    try:
        info = rag.client.get_collection(collections[0])
        if info.points_count == 0:
            pytest.skip("No documents ingested yet")
    except Exception:
        pytest.skip("Collections not available")
    
    result = rag.answer_question(
        "What is a reasonable accommodation?",
        stream=False
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_search_with_summary_gating(rag):
    query = "what are some tips for job interviews for people with autism?"
    
    context_docs = rag.search_with_summary_gating(query)
    
    assert context_docs is not None
    assert isinstance(context_docs, list)
    assert len(context_docs) > 0
    
    for doc in context_docs[:3]:
        assert isinstance(doc, dict)
        assert 'source_path' in doc or 'payload' in doc
        assert 'score' in doc
        
        if 'payload' in doc:
            payload = doc['payload']
            assert 'source_path' in payload
        else:
            assert 'text' in doc or 'chunk_text' in doc


def test_answer_question_with_summary_gating(rag):
    query = "what are some tips for job interviews for people with autism?"
    
    result = rag.answer_question(
        query,
        stream=False,
        use_summary_gating=True,
        use_parent_docs=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 100
    assert "interview" in result.lower() or "job" in result.lower()


def test_streaming_answer(rag):
    query = "what are tips for job interviews?"
    
    result_stream = rag.answer_question(
        query,
        stream=True,
        use_summary_gating=True,
        use_parent_docs=True
    )
    
    full_response = ""
    chunk_count = 0
    for chunk in result_stream:
        full_response += chunk
        chunk_count += 1
    
    assert chunk_count > 0
    assert len(full_response) > 100
    assert isinstance(full_response, str)
