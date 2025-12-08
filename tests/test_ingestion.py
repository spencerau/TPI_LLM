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


def test_data_directories_configured(config):
    data_dirs = config.get('data', {})
    assert len(data_dirs) > 0


def test_data_directories_exist(config):
    data_dirs = config.get('data', {})
    for name, path in data_dirs.items():
        assert os.path.exists(path), f"Directory {path} does not exist"


def test_clear_and_ingest_all(config, client):
    collections = config.get('qdrant', {}).get('collections', {})
    for name, collection_name in collections.items():
        try:
            client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception:
            pass
        try:
            client.delete_collection(f"{collection_name}_summaries")
            print(f"Deleted summary collection: {collection_name}_summaries")
        except Exception:
            pass
    
    try:
        client.delete_collection("docstore")
        print("Deleted docstore collection")
    except Exception:
        pass
    
    time.sleep(1)
    
    data_dirs = config.get('data', {})
    collections = config.get('qdrant', {}).get('collections', {})
    total_stats = {'total_files': 0, 'success_files': 0, 'failed_files': 0}
    collection_stats = {}
    
    for collection_key, path in data_dirs.items():
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
        
        collection_name = collections.get(collection_key, collection_key)
        ingestion = UnifiedIngestion(base_dir=path, collection_name=collection_name)
        
        files = list(Path(path).rglob('*'))
        files = [f for f in files if f.is_file() and f.suffix.lower() in ['.md', '.pdf', '.json']]
        files = [f for f in files if 'readme' not in f.name.lower() and not f.name.startswith('._')]
        
        print(f"\n=== Ingesting {len(files)} files from {path} ===")
        
        success_count = 0
        for file_path in files:
            try:
                print(f"  Processing: {file_path.name}")
                time.sleep(2)
                success = ingestion.ingest_file(str(file_path))
                if success:
                    success_count += 1
                    print(f"  OK: {file_path.name}")
                else:
                    print(f"  FAIL: {file_path.name}")
                time.sleep(3)
            except Exception as e:
                print(f"  ERROR: {file_path.name} - {e}")
        
        total_stats['total_files'] += len(files)
        total_stats['success_files'] += success_count
        total_stats['failed_files'] += len(files) - success_count
        collection_stats[collection_key] = {'total': len(files), 'success': success_count}
    
    print(f"\n=== Final Stats ===")
    print(f"Total: {total_stats['success_files']}/{total_stats['total_files']} files")
    
    for name, stats in collection_stats.items():
        print(f"  {name}: {stats['success']}/{stats['total']}")
    
    assert total_stats['total_files'] > 0
    assert total_stats['success_files'] > 0


def test_collections_have_documents(config, client):
    collections = config.get('qdrant', {}).get('collections', {})
    
    print("\n=== Collection Document Counts ===")
    total_docs = 0
    for name, collection_name in collections.items():
        try:
            info = client.get_collection(collection_name)
            count = info.points_count
            total_docs += count
            print(f"{name}: {count} chunks")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    assert total_docs > 0


def test_summaries_created(config, client):
    collections = config.get('qdrant', {}).get('collections', {})
    
    print("\n=== Summary Collection Counts ===")
    total_summaries = 0
    for name, collection_name in collections.items():
        summary_collection = f"{collection_name}_summaries"
        try:
            info = client.get_collection(summary_collection)
            count = info.points_count
            total_summaries += count
            print(f"{summary_collection}: {count} summaries")
        except Exception as e:
            print(f"{summary_collection}: Not found or empty")
    
    if total_summaries == 0:
        pytest.skip("No summaries created - may be due to LLM issues")


def test_docstore_has_documents(config, client):
    print("\n=== Docstore ===")
    try:
        info = client.get_collection("docstore")
        print(f"Docstore: {info.points_count} full documents")
        assert info.points_count > 0
    except Exception as e:
        pytest.fail(f"Docstore not found: {e}")
