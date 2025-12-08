import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_rag.ingestion import UnifiedIngestion
from core_rag.utils import load_config


def main():
    print("Starting TPI document ingestion...")
    
    config = load_config()
    data_dirs = config.get('data', {})
    
    if not data_dirs:
        print("No data directories configured in config.yaml")
        return
    
    ingestion = UnifiedIngestion()
    
    total_stats = {
        'total_files': 0,
        'success_files': 0,
        'failed_files': 0
    }
    
    for name, path in data_dirs.items():
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            continue
            
        print(f"\nIngesting {name} from {path}...")
        stats = ingestion.ingest_directory(path)
        
        total_stats['total_files'] += stats.get('total_files', 0)
        total_stats['success_files'] += stats.get('success_files', 0)
        total_stats['failed_files'] += stats.get('failed_files', 0)
        
        print(f"  Total: {stats.get('total_files', 0)}, "
              f"Success: {stats.get('success_files', 0)}, "
              f"Failed: {stats.get('failed_files', 0)}")
    
    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"  Total files: {total_stats['total_files']}")
    print(f"  Successful: {total_stats['success_files']}")
    print(f"  Failed: {total_stats['failed_files']}")
    
    ingestion.print_collection_summary()


if __name__ == "__main__":
    main()
