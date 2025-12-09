import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_rag.retrieval import UnifiedRAG
from core_rag.ingestion import UnifiedIngestion
from core_rag.utils import load_config


def init_session_state():
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


def init_rag():
    try:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag = UnifiedRAG()
            st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False


def get_collection_stats():
    if not st.session_state.rag:
        return {}
    
    stats = {}
    config = load_config()
    collections = config.get('qdrant', {}).get('collections', {})
    
    for name, collection_name in collections.items():
        try:
            scroll_result = st.session_state.rag.client.scroll(
                collection_name="docstore",
                scroll_filter={
                    "must": [
                        {
                            "key": "collection_name",
                            "match": {"value": collection_name}
                        }
                    ]
                },
                limit=10000,
                with_payload=True
            )
            
            parent_doc_count = len(scroll_result[0]) if scroll_result else 0
            
            stats[name] = {
                'collection_name': collection_name,
                'points_count': parent_doc_count,
                'status': 'active'
            }
        except Exception as e:
            stats[name] = {
                'collection_name': collection_name,
                'points_count': 0,
                'status': f'error: {str(e)}'
            }
    return stats


def run_query(query: str, use_streaming: bool = True):
    if not st.session_state.rag:
        return "RAG system not initialized", [], None
    
    try:
        result = st.session_state.rag.answer_question(
            query,
            stream=use_streaming,
            selected_collections=None,
            use_parent_docs=True,
            use_summary_gating=True,
            return_debug_info=True,
            enable_thinking=True,
            show_thinking=False
        )
        
        if use_streaming:
            gen, context_docs, debug = result
            selected_collections = debug.get('collections_searched', [])
            return gen, context_docs, selected_collections
        else:
            answer, context_docs, debug = result
            selected_collections = debug.get('collections_searched', [])
            return answer, context_docs, selected_collections
    except Exception as e:
        import traceback
        st.error(f"Query error: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}", [], None

def search_collection(query: str, collection_name: str, top_k: int = 5):
    if not st.session_state.rag:
        return []
    
    try:
        results = st.session_state.rag.search_collection(
            query=query,
            collection_name=collection_name,
            top_k=top_k
        )
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def ingest_documents(directory: str):
    try:
        with st.spinner(f"Ingesting documents from {directory}..."):
            ingestion = UnifiedIngestion()
            stats = ingestion.ingest_directory(directory)
            return stats
    except Exception as e:
        st.error(f"Ingestion error: {str(e)}")
        return None


def render_sidebar():
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("Initialize RAG System", type="primary"):
            if init_rag():
                st.success("RAG system initialized!")
            
        if st.session_state.initialized:
            st.success("RAG System Active")
        else:
            st.warning("RAG System Not Initialized")
        
        st.divider()
        
        st.header("Collection Statistics")
        if st.button("Refresh Stats"):
            pass
        
        if st.session_state.initialized:
            stats = get_collection_stats()
            for name, info in stats.items():
                with st.expander(f"{name}"):
                    st.write(f"**Collection:** {info['collection_name']}")
                    st.write(f"**Documents:** {info['points_count']}")
                    st.write(f"**Status:** {info['status']}")
        
        st.divider()
        
        st.header("Document Ingestion")
        ingest_dir = st.text_input(
            "Directory to ingest:",
            value="tpi_documents",
            help="Relative path from project root"
        )
        
        if st.button("Ingest Documents"):
            if ingest_dir:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                full_path = os.path.join(project_root, ingest_dir)
                
                if os.path.exists(full_path):
                    result = ingest_documents(full_path)
                    if result:
                        st.success(f"Ingested! Total: {result.get('total_files', 0)}, "
                                   f"Success: {result.get('success_files', 0)}, "
                                   f"Failed: {result.get('failed_files', 0)}")
                else:
                    st.error(f"Directory not found: {full_path}")


def render_source_documents(context_docs, message_index=None):
    if not context_docs:
        return
    
    unique_docs = {}
    for doc in context_docs:
        if isinstance(doc, dict):
            if 'payload' in doc:
                payload = doc['payload']
                source_path = payload.get('source_path', 'Unknown')
                doc_id = payload.get('parent_doc_id') or payload.get('doc_id', 'N/A')
                title = payload.get('title', source_path)
                text = payload.get('full_text') or payload.get('chunk_text', '')
                score = doc.get('score', 0)
            else:
                metadata = doc.get('metadata', {})
                source_path = doc.get('source_path') or metadata.get('source_path', 'Unknown')
                doc_id = doc.get('doc_id') or metadata.get('doc_id', 'N/A')
                title = doc.get('title') or metadata.get('title', source_path)
                text = doc.get('text', '')
                score = doc.get('score', 0)
            
            key = doc_id if doc_id != 'N/A' else source_path
            if key not in unique_docs:
                unique_docs[key] = {
                    'score': score,
                    'path': source_path,
                    'doc_id': doc_id,
                    'title': title,
                    'text': text
                }
    
    expander_key = f"sources_{message_index}" if message_index is not None else None
    with st.expander(f"View {len(unique_docs)} Source Documents"):
        for i, (key, doc_info) in enumerate(unique_docs.items(), 1):
            st.markdown(f"**Source {i}**: {doc_info['title']}")
            st.caption(f"Score: {doc_info['score']:.4f}")
            st.caption(f"Path: {doc_info['path']}")
            st.caption(f"Document ID: {doc_info['doc_id']}")
            
            text_key = f"source_{i}_{message_index}" if message_index else f"source_{i}"
            st.text_area(
                f"Content {i}",
                doc_info['text'][:3000],
                height=200,
                key=text_key
            )
            st.divider()


def render_chat_tab():
    st.header("Chat with TPI Documents")
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
            if message["role"] == "assistant":
                if "sources" in message and message["sources"]:
                    render_source_documents(message["sources"], message_index=idx)
                
                if "collections" in message and message["collections"]:
                    st.caption(f"Searched collections: {', '.join(message['collections'])}")
    
    if prompt := st.chat_input("Ask a question about TPI employment resources..."):
        if not st.session_state.initialized:
            st.warning("Please initialize the RAG system first using the sidebar button.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Routing query and retrieving documents..."):
                response_gen, context_docs, collections = run_query(prompt, use_streaming=True)
            
            with st.spinner("Thinking..."):
                full_response = ""
                for chunk in response_gen:
                    full_response += chunk
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": context_docs,
                "collections": collections
            })
            st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def render_info_tab():
    st.header("System Information")
    
    config = load_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.json({
            "qdrant_host": config.get('qdrant', {}).get('host', 'N/A'),
            "qdrant_port": config.get('qdrant', {}).get('port', 'N/A'),
            "embedding_model": config.get('embedding', {}).get('model', 'N/A'),
            "llm_model": config.get('llm', {}).get('primary_model', 'N/A'),
            "ollama_host": config.get('embedding', {}).get('ollama_host', 'N/A'),
        })
    
    with col2:
        st.subheader("Collections")
        collections = config.get('qdrant', {}).get('collections', {})
        for name, collection in collections.items():
            st.write(f"**{name}**: `{collection}`")
    
    st.subheader("Quick Start")
    st.markdown("""
    1. Initialize: Click Initialize RAG System in the sidebar
    2. Ingest: Use the ingestion section to load documents (if needed)
    3. Chat: Ask questions in the Chat tab
    """)


def run_app():
    st.set_page_config(
        page_title="TPI RAG Assistant",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("TPI RAG Assistant")
    st.markdown("Test interface for the Core RAG pipeline with TPI employment documents.")
    
    render_sidebar()
    
    tab1, tab2 = st.tabs(["Chat", "Info"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_info_tab()
