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


def test_no_output_truncation(rag):
    """Test that LLM responses are not prematurely truncated"""
    query = "What are some tips for preparing for job interviews?"
    
    result = rag.answer_question(
        query,
        stream=False,
        use_summary_gating=False,
        use_parent_docs=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 150, f"Response too short ({len(result)} chars), may be truncated: {result}"
    
    assert not result.endswith('|'), "Response ends with '|' - possible truncation"
    
    assert "interview" in result.lower() or "tip" in result.lower() or "prepare" in result.lower(), \
        "Response doesn't contain relevant keywords - may be incomplete"
    
    result_stream = rag.answer_question(
        query,
        stream=True,
        use_summary_gating=False,
        use_parent_docs=True
    )
    
    streamed_response = "".join(result_stream)
    assert len(streamed_response) > 150, f"Streamed response too short ({len(streamed_response)} chars)"
    assert not streamed_response.endswith('|'), "Streamed response ends with '|' - possible truncation"


def test_application_interview_collection(rag):
    """Test querying application_interview_process collection with parent docs"""
    query = "What should I do to prepare for a job interview?"
    
    results = rag.search_collection(
        query=query,
        collection_name="application_interview_process",
        top_k=10
    )
    
    assert len(results) > 0, "No results from application_interview_process collection"
    
    answer = rag.answer_question(
        query,
        stream=False,
        use_summary_gating=True,
        use_parent_docs=True
    )
    
    assert len(answer) > 100, f"Answer too short: {len(answer)} chars"
    assert not answer.endswith('|'), "Answer appears truncated"
    assert "interview" in answer.lower() or "prepare" in answer.lower()


def test_legal_policy_collection(rag):
    """Test querying legal_policy collection with parent docs"""
    query = "What are the legal requirements for accessible interviews?"
    
    results = rag.search_collection(
        query=query,
        collection_name="legal_policy",
        top_k=10
    )
    
    assert len(results) > 0, "No results from legal_policy collection"
    
    answer = rag.answer_question(
        query,
        stream=False,
        selected_collections=["legal_policy"],
        use_summary_gating=True,
        use_parent_docs=True
    )
    
    assert len(answer) > 100, f"Answer too short: {len(answer)} chars"
    assert not answer.endswith('|'), "Answer appears truncated"


def test_best_practices_collection(rag):
    """Test querying best_practices collection with parent docs"""
    query = "What are best practices for workplace accommodations and assistive technology?"
    
    results = rag.search_collection(
        query=query,
        collection_name="best_practices",
        top_k=10
    )
    
    assert len(results) > 0, "No results from best_practices collection"
    
    answer = rag.answer_question(
        query,
        stream=False,
        selected_collections=["best_practices"],
        use_summary_gating=True,
        use_parent_docs=True
    )
    
    assert len(answer) > 100, f"Answer too short: {len(answer)} chars"
    assert not answer.endswith('|'), "Answer appears truncated"
    assert "accommodation" in answer.lower() or "workplace" in answer.lower() or "technology" in answer.lower()


def test_workplace_partnerships_collection(rag):
    """Test querying workplace_partnerships collection with parent docs"""
    query = "How can employers engage with workforce development programs?"
    
    results = rag.search_collection(
        query=query,
        collection_name="workplace_partnerships",
        top_k=10
    )
    
    assert len(results) > 0, "No results from workplace_partnerships collection"
    
    answer = rag.answer_question(
        query,
        stream=False,
        selected_collections=["workplace_partnerships"],
        use_summary_gating=True,
        use_parent_docs=True
    )
    
    assert len(answer) > 100, f"Answer too short: {len(answer)} chars"
    assert not answer.endswith('|'), "Answer appears truncated"
    assert "employer" in answer.lower() or "engagement" in answer.lower() or "workforce" in answer.lower()


def test_evaluation_collection(rag):
    """Test querying evaluation collection with parent docs"""
    query = "What are the employment skills development criteria?"
    
    results = rag.search_collection(
        query=query,
        collection_name="evaluation",
        top_k=10
    )
    
    assert len(results) > 0, "No results from evaluation collection"
    
    answer = rag.answer_question(
        query,
        stream=False,
        selected_collections=["evaluation"],
        use_summary_gating=True,
        use_parent_docs=True
    )
    
    assert len(answer) > 100, f"Answer too short: {len(answer)} chars"
    assert not answer.endswith('|'), "Answer appears truncated"
    assert "skill" in answer.lower() or "employment" in answer.lower() or "development" in answer.lower()


def test_parent_docs_retrieval(rag):
    """Test that parent documents are properly retrieved and complete"""
    query = "Tell me about job interview preparation"
    
    context_docs = rag.search_with_summary_gating(query)
    
    assert len(context_docs) > 0, "No context documents retrieved"
    
    for doc in context_docs[:3]:
        if 'parent_text' in doc:
            parent_text = doc['parent_text']
            assert len(parent_text) > 500, f"Parent doc too short: {len(parent_text)} chars - may not be full document"
            assert not parent_text.endswith('|'), "Parent doc appears truncated"
        elif 'text' in doc:
            text = doc['text']
            assert len(text) > 100, f"Doc text too short: {len(text)} chars"

