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
            info = st.session_state.rag.client.get_collection(collection_name)
            stats[name] = {
                'collection_name': collection_name,
                'points_count': info.points_count,
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
        return "RAG system not initialized", []
    
    try:
        context_docs = st.session_state.rag.search_with_summary_gating(query)
        
        result = st.session_state.rag.answer_question(
            query,
            stream=use_streaming,
            use_parent_docs=True,
            use_summary_gating=True
        )
        
        if use_streaming:
            full_response = ""
            for chunk in result:
                full_response += chunk
            return full_response, context_docs
        else:
            return result, context_docs
    except Exception as e:
        return f"Error: {str(e)}", []


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
                text = payload.get('full_text') or payload.get('chunk_text', '')
                score = doc.get('score', 0)
            else:
                source_path = doc.get('source_path', 'Unknown')
                doc_id = doc.get('doc_id', 'N/A')
                text = doc.get('text', '')
                score = doc.get('score', 0)
            
            if source_path not in unique_docs:
                unique_docs[source_path] = {
                    'score': score,
                    'path': source_path,
                    'doc_id': doc_id,
                    'text': text
                }
    
    expander_key = f"sources_{message_index}" if message_index is not None else None
    with st.expander(f"View {len(unique_docs)} Source Documents"):
        for i, (path, doc_info) in enumerate(unique_docs.items(), 1):
            st.markdown(f"**Source {i}** (Score: {doc_info['score']:.4f})")
            st.caption(f"{doc_info['path']}")
            st.caption(f"Document ID: {doc_info['doc_id']}")
            
            key = f"source_{i}_{message_index}" if message_index else f"source_{i}"
            st.text_area(
                f"Content {i}",
                doc_info['text'][:3000],
                height=200,
                key=key
            )
            st.divider()


def render_chat_tab():
    st.header("Chat with TPI Documents")
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                render_source_documents(message["sources"], message_index=idx)
    
    if prompt := st.chat_input("Ask a question about TPI employment resources..."):
        if not st.session_state.initialized:
            st.warning("Please initialize the RAG system first using the sidebar button.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, context_docs = run_query(prompt)
                st.markdown(response, unsafe_allow_html=True)
                render_source_documents(context_docs, message_index=len(st.session_state.messages))
            
            st.session_state.messages.append({"role": "assistant", "content": response, "sources": context_docs})
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def render_search_tab():
    st.header("Search Collections")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search query:", placeholder="Enter your search...")
    
    with col2:
        config = load_config()
        collections = config.get('qdrant', {}).get('collections', {})
        collection_options = list(collections.values())
        selected_collection = st.selectbox("Collection:", collection_options)
    
    top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    
    if st.button("Search", type="primary"):
        if search_query and st.session_state.initialized:
            results = search_collection(search_query, selected_collection, top_k)
            
            if results:
                st.subheader(f"Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} (Score: {result.get('score', 'N/A'):.4f})"):
                        if 'chunk_text' in result.get('payload', {}):
                            st.markdown(result['payload']['chunk_text'])
                        if 'source_path' in result.get('payload', {}):
                            st.caption(f"Source: {result['payload']['source_path']}")
            else:
                st.info("No results found")
        elif not st.session_state.initialized:
            st.warning("Please initialize the RAG system first.")


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
    4. Search: Use the Search tab to explore specific collections
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
    
    tab1, tab2, tab3 = st.tabs(["Chat", "Search", "Info"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_search_tab()
    
    with tab3:
        render_info_tab()
