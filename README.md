# TPI_LLM
Job Training / Job Coach for both employees and employers within the context of disabilities. Created in conjunction with the Thompson Policy Institute at Chapman University.

## Getting Started

1. Create a virtual environment, activate, and install dependencies:
- `python3.12 -m venv .venv`
- `source .venv/bin/acticate`
- `pip install -r requirements.txt`
<br>

2. Placeholder here about installing public ssh keys onto compute cluster
<br>

3. Run ingestion and web UI
- `python src/ingest.py`
- `streamlit run src/streamlit_ui.py`
<br>

4. Run tests:
- `pytest -q`

## Project Layout
- `src/` — application and ingestion code
- `tests/` — unit and integration tests
- `tpi_documents/` — sample documents used for ingestion and RAG
- `configs/` — configuration (e.g. `config.yaml`)
- `requirements.txt`, `Dockerfile`, `docker-compose.yml` — environment and containerization

## TODO: 
- Memory feature to save chat history (prob need to update Core_RAG for this)
<br>

- Switch to vLLM (or SGLang?) instead of using Ollama
<br>

- Switch Document ingestion to use DocLing (or Deepseek-OCR-2?)
<br>

- Pull from an online source for documents for ingestion pipeline
<br>

- Unsloth AI (or another fine tuning framework) to finetune model to data?