# School Regulation RAG — E5 + BM25 + Ollama

## Stack

- Dense Retriever: fine-tuned `intfloat/multilingual-e5-base`
- Sparse Retriever: BM25S
- Fusion: Reciprocal Rank Fusion
- Generator: Ollama + `qwen2.5:3b`
- Backend: FastAPI
- Database: Supabase PostgreSQL + pgvector

## Setup

Copy artifacts from notebook 06 into:

```text
artifacts/hybrid_retriever_e5/
```

Install and start Ollama:

```bash
ollama serve
ollama pull qwen2.5:3b
```

Install backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

API docs:

```text
http://localhost:8000/docs
```

Endpoints:

```text
GET  /api/v1/health
POST /api/v1/retrieve
POST /api/v1/ask
```

When FastAPI runs in Docker and Ollama runs on the host, set:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```
