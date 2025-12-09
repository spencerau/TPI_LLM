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


def test_router_initialization(rag, config):
    """Test that query router is properly initialized"""
    assert rag.query_router is not None
    router_config = config.get('query_router', {})
    assert 'collection_descriptions' in router_config
    assert len(router_config['collection_descriptions']) == 5


def test_application_interview_routing(rag):
    """Test queries about job applications and interviews route correctly"""
    queries = [
        "How do I prepare for a job interview?",
        "What should be in my resume as a person with a disability?",
        "How do I request accommodations during the interview process?",
        "What are accessible job description best practices?",
        "Tips for disclosing my disability during job application"
    ]
    
    for query in queries:
        result = rag.query_router.route_query(query)
        assert 'collections' in result
        assert 'application_interview_process' in result['collections'], \
            f"Query '{query}' should route to application_interview_process but got {result['collections']}"


def test_legal_policy_routing(rag):
    """Test queries about legal and policy matters route correctly"""
    queries = [
        "What are the ADA requirements for employers?",
        "What does Section 504 require?",
        "What are reasonable accommodation legal requirements?",
        "California FEHA employment law",
        "WIOA policy guidance for disability employment"
    ]
    
    for query in queries:
        result = rag.query_router.route_query(query)
        assert 'collections' in result
        assert 'legal_policy' in result['collections'], \
            f"Query '{query}' should route to legal_policy but got {result['collections']}"


def test_best_practices_routing(rag):
    """Test queries about workplace best practices route correctly"""
    queries = [
        "What are common workplace accommodations?",
        "How do I onboard an employee with a disability?",
        "What assistive technology is available?",
        "Best practices for inclusive supervision",
        "Job Accommodation Network resources",
        "How to provide feedback to employees with disabilities"
    ]
    
    for query in queries:
        result = rag.query_router.route_query(query)
        assert 'collections' in result
        assert 'best_practices' in result['collections'], \
            f"Query '{query}' should route to best_practices but got {result['collections']}"


def test_workplace_partnerships_routing(rag):
    """Test queries about partnerships and community collaboration route correctly"""
    queries = [
        "How can employers engage with workforce development programs?",
        "What is the role of Regional Centers in employment?",
        "Sample MOU for disability employment partnerships",
        "Job coaching and supported employment models",
        "Employer recruitment strategies for disability inclusion"
    ]
    
    for query in queries:
        result = rag.query_router.route_query(query)
        assert 'collections' in result
        assert 'workplace_partnerships' in result['collections'], \
            f"Query '{query}' should route to workplace_partnerships but got {result['collections']}"
    
    wotc_result = rag.query_router.route_query("Work Opportunity Tax Credit for hiring people with disabilities")
    assert 'workplace_partnerships' in wotc_result['collections'] or 'best_practices' in wotc_result['collections'], \
        "WOTC query should route to workplace_partnerships or best_practices"


def test_evaluation_routing(rag):
    """Test queries about evaluation and metrics route correctly"""
    queries = [
        "How do I measure employment outcomes?",
        "What are good evaluation metrics for disability employment?",
        "Employment skills development assessment",
        "How to conduct accessibility audits?",
        "Survey tools for employer feedback"
    ]
    
    for query in queries:
        result = rag.query_router.route_query(query)
        assert 'collections' in result
        assert 'evaluation' in result['collections'], \
            f"Query '{query}' should route to evaluation but got {result['collections']}"
    
    pre_emp_result = rag.query_router.route_query("Pre-employment skills evaluation criteria")
    collections = pre_emp_result['collections']
    assert 'evaluation' in collections or 'application_interview_process' in collections, \
        "Pre-employment skills should route to evaluation or application_interview_process"


def test_multi_collection_routing(rag):
    """Test that complex queries can route to multiple relevant collections"""
    query = "What are the legal requirements and best practices for workplace accommodations?"
    
    result = rag.query_router.route_query(query)
    assert 'collections' in result
    collections = result['collections']
    
    assert len(collections) >= 2, "Complex query should route to multiple collections"
    assert 'legal_policy' in collections or 'best_practices' in collections, \
        "Query about legal requirements and best practices should route to relevant collections"


def test_router_token_allocation(rag):
    """Test that router properly allocates tokens"""
    query = "How do I prepare for a job interview?"
    
    result = rag.query_router.route_query(query)
    assert 'token_allocation' in result
    assert isinstance(result['token_allocation'], int)
    assert result['token_allocation'] > 0


def test_router_returns_collections_list(rag):
    """Test that router returns a list of collection names"""
    query = "Tell me about disability employment"
    
    result = rag.query_router.route_query(query)
    assert 'collections' in result
    assert isinstance(result['collections'], list)
    assert len(result['collections']) > 0
    
    valid_collections = {
        'application_interview_process',
        'legal_policy',
        'best_practices',
        'workplace_partnerships',
        'evaluation'
    }
    for collection in result['collections']:
        assert collection in valid_collections, f"Invalid collection returned: {collection}"


def test_router_handles_ambiguous_query(rag):
    """Test that router handles ambiguous queries gracefully"""
    query = "Tell me about employment"
    
    result = rag.query_router.route_query(query)
    assert 'collections' in result
    assert len(result['collections']) > 0, "Router should return at least one collection for ambiguous query"


def test_router_empty_query(rag):
    """Test router behavior with empty query"""
    query = ""
    
    result = rag.query_router.route_query(query)
    assert 'collections' in result
    assert isinstance(result['collections'], list)


def test_collection_descriptions_coverage(config):
    """Test that all configured collections have descriptions"""
    router_config = config.get('query_router', {})
    collection_descriptions = router_config.get('collection_descriptions', {})
    default_collections = router_config.get('default_collections', [])
    
    for collection in default_collections:
        assert collection in collection_descriptions, \
            f"Collection '{collection}' is in default_collections but missing description"
        assert collection_descriptions[collection], \
            f"Collection '{collection}' has empty description"


def test_specific_topic_routing_accuracy(rag):
    """Test routing accuracy for specific topics from collection descriptions"""
    
    assert 'application_interview_process' in rag.query_router.route_query(
        "accessible job descriptions examples"
    )['collections']
    
    assert 'legal_policy' in rag.query_router.route_query(
        "ADA employer obligations"
    )['collections']
    
    assert 'best_practices' in rag.query_router.route_query(
        "assistive technology resources"
    )['collections']
    
    assert 'workplace_partnerships' in rag.query_router.route_query(
        "supported employment models"
    )['collections']
    
    assert 'evaluation' in rag.query_router.route_query(
        "outcome metrics for placement and retention"
    )['collections']
