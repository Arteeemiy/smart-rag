# smart-rag

Production-style FastAPI RAG service template with Chroma, HuggingFace embeddings, and an OpenAI-compatible chat API.

## Stack
- FastAPI
- Poetry
- ChromaDB
- sentence-transformers / HuggingFace embeddings
- requests-based LLM client

## Project structure
- `src/smart_rag/main.py` — FastAPI app assembly
- `src/smart_rag/__main__.py` — package entrypoint for `python -m smart_rag`
- `src/smart_rag/api/v1/routes.py` — HTTP routes
- `src/smart_rag/services/` — application business logic
- `src/smart_rag/clients/` — external clients (LLM, vector store)
- `scripts/build_index.py` — index builder

## Run locally
Set `LLM_API_URL` and `LLM_MODEL` in `.env` before running.

```bash
cp .env.example .env
poetry install
poetry run python scripts/build_index.py
poetry run uvicorn smart_rag.main:app --reload
```

Alternative run mode:
```bash
poetry run python -m smart_rag
```

## Endpoints
- `GET /health`
- `GET /ready`
- `GET /api/v1/ping`
- `POST /api/v1/rag/ask`
- `POST /api/v1/rag/retrieve`
