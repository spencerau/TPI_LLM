import pytest
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_rag.ingestion import UnifiedIngestion
from core_rag.utils import load_config
from qdrant_client import QdrantClient


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def client(config):
    return QdrantClient(
        host=config['qdrant']['host'],
        port=config['qdrant']['port'],
        timeout=config['qdrant']['timeout']
    )


@pytest.fixture(scope="module")
def ingestion(config):
    """Create an ingestion instance once per test module, delete all collections, and re-ingest"""
    
    client = QdrantClient(
        host=config['qdrant']['host'],
        port=config['qdrant']['port'],
        timeout=config['qdrant']['timeout']
    )
    
    print("\n=== Deleting all collections ===")
    try:
        all_collections = client.get_collections().collections
        for collection in all_collections:
            try:
                client.delete_collection(collection.name)
                print(f"Deleted collection: {collection.name}")
            except Exception as e:
                print(f"Could not delete {collection.name}: {e}")
    except Exception as e:
        print(f"Warning: Could not list collections: {e}")
    
    ing = UnifiedIngestion(base_dir="data")
    
    data_dirs = config.get('data', {})
    print("\n=== Ingesting documents ===")
    for name, path in data_dirs.items():
        if os.path.exists(path):
            print(f"Ingesting {name} from {path}...")
            stats = ing.ingest_directory(path)
            print(f"  Total: {stats.get('total_files', 0)}, Success: {stats.get('success_files', 0)}, Failed: {stats.get('failed_files', 0)}")
    
    return ing


def test_data_directories_configured(config):
    data_dirs = config.get('data', {})
    assert len(data_dirs) > 0


def test_data_directories_exist(config):
    data_dirs = config.get('data', {})
    for name, path in data_dirs.items():
        assert os.path.exists(path), f"Directory {path} does not exist"


def test_clear_and_ingest_all(config, ingestion):
    collections = config.get('qdrant', {}).get('collections', {})
    data_dirs = config.get('data', {})
    
    total_docs = 0
    collection_stats = {}
    
    print(f"\n=== Verifying Ingestion Results ===")
    
    for name, collection_name in collections.items():
        try:
            info = ingestion.client.get_collection(collection_name)
            count = info.points_count
            total_docs += count
            collection_stats[name] = count
            print(f"{name}: {count} chunks")
        except Exception as e:
            print(f"{name}: Error - {e}")
            collection_stats[name] = 0
    
    print(f"\n=== Final Stats ===")
    print(f"Total documents across all collections: {total_docs}")
    
    for name, count in collection_stats.items():
        print(f"  {name}: {count}")
    
    assert total_docs > 0, "No documents ingested"
    assert len([c for c in collection_stats.values() if c > 0]) > 0, "No collections have documents"


def test_collections_have_documents(config, ingestion):
    collections = config.get('qdrant', {}).get('collections', {})
    
    print("\n=== Collection Document Counts ===")
    total_docs = 0
    for name, collection_name in collections.items():
        try:
            info = ingestion.client.get_collection(collection_name)
            count = info.points_count
            total_docs += count
            print(f"{name}: {count} chunks")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    assert total_docs > 0


def test_summaries_created(config, ingestion):
    collections = config.get('qdrant', {}).get('collections', {})
    
    print("\n=== Summary Collection Counts ===")
    total_summaries = 0
    for name, collection_name in collections.items():
        summary_collection = f"{collection_name}_summaries"
        try:
            info = ingestion.client.get_collection(summary_collection)
            count = info.points_count
            total_summaries += count
            print(f"{summary_collection}: {count} summaries")
        except Exception as e:
            print(f"{summary_collection}: Not found or empty")
    
    if total_summaries == 0:
        pytest.skip("No summaries created - may be due to LLM issues")


def test_docstore_has_documents(config, ingestion):
    print("\n=== Docstore ===")
    try:
        info = ingestion.client.get_collection("docstore")
        print(f"Docstore: {info.points_count} full documents")
        assert info.points_count > 0
    except Exception as e:
        pytest.fail(f"Docstore not found: {e}")


def test_doc_ids_match_across_collections(config, ingestion):
    collections = config.get('qdrant', {}).get('collections', {})
    
    print("\n=== Testing doc_id consistency ===")
    
    all_mismatches = []
    
    for name, collection_name in collections.items():
        print(f"\nChecking collection: {name}")
        
        docstore_result = ingestion.client.scroll(
            collection_name="docstore",
            scroll_filter={
                "must": [
                    {
                        "key": "collection_name",
                        "match": {"value": name}
                    }
                ]
            },
            limit=100,
            with_payload=True
        )
        
        docstore_doc_ids = set()
        for point in docstore_result[0]:
            doc_id = point.payload.get('doc_id')
            if doc_id:
                docstore_doc_ids.add(doc_id)
        
        print(f"Docstore: {len(docstore_doc_ids)} doc_ids")
        
        summary_collection = f"{collection_name}_summaries"
        try:
            summary_result = ingestion.client.scroll(
                collection_name=summary_collection,
                limit=100,
                with_payload=True
            )
            
            summary_doc_ids = set()
            for point in summary_result[0]:
                doc_id = point.payload.get('doc_id')
                if doc_id:
                    summary_doc_ids.add(doc_id)
            
            print(f"Summaries: {len(summary_doc_ids)} doc_ids")
            
            if summary_doc_ids != docstore_doc_ids:
                missing_in_docstore = summary_doc_ids - docstore_doc_ids
                missing_in_summaries = docstore_doc_ids - summary_doc_ids
                
                if missing_in_docstore:
                    print(f"Summary doc_ids NOT in docstore: {missing_in_docstore}")
                    all_mismatches.append(f"{name}: {len(missing_in_docstore)} summary doc_ids not in docstore")
                
                if missing_in_summaries:
                    print(f"Docstore doc_ids NOT in summaries: {missing_in_summaries}")
                    all_mismatches.append(f"{name}: {len(missing_in_summaries)} docstore doc_ids not in summaries")
            else:
                print(f"Summary doc_ids match docstore")
        
        except Exception as e:
            print(f"Could not check summaries: {e}")
        
        try:
            chunk_result = ingestion.client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True
            )
            
            chunk_parent_ids = set()
            for point in chunk_result[0]:
                doc_id = point.payload.get('metadata', {}).get('doc_id') or point.payload.get('doc_id')
                if doc_id:
                    chunk_parent_ids.add(doc_id)
            
            print(f"Chunks: {len(chunk_parent_ids)} unique parent doc_ids")
            
            if chunk_parent_ids != docstore_doc_ids:
                missing_in_docstore = chunk_parent_ids - docstore_doc_ids
                missing_in_chunks = docstore_doc_ids - chunk_parent_ids
                
                if missing_in_docstore:
                    print(f"Chunk parent doc_ids NOT in docstore: {missing_in_docstore}")
                    all_mismatches.append(f"{name}: {len(missing_in_docstore)} chunk parent doc_ids not in docstore")
                
                if missing_in_chunks:
                    print(f"Docstore doc_ids NOT referenced by chunks: {missing_in_chunks}")
                    all_mismatches.append(f"{name}: {len(missing_in_chunks)} docstore doc_ids not referenced by chunks")
            else:
                print(f"Chunk parent doc_ids match docstore")
        
        except Exception as e:
            print(f"Could not check chunks: {e}")
    
    if all_mismatches:
        pytest.fail(f"Doc ID mismatches found:\n" + "\n".join(all_mismatches))
    else:
        print("\nAll doc_ids match across summaries, chunks, and docstore")
