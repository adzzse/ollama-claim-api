# AGENTS.md

## What this is

FastAPI demo (v0.2.0) for evidence-based claim analysis using local Ollama/Qwen models. In-memory store — restarting uvicorn clears all data.

## Two code paths (do not conflate)

- **`app/main.py`** — FastAPI app with endpoints: `/health`, `/process/claim`, `/sources`, `/match/claim`, `/papers`, `/review/paper`, `/ai/embeddings`.
- **`ai_worker.py`** — separate background worker that reads from RabbitMQ, downloads PDFs from MinIO, extracts+chunks+embeds, publishes results. NOT part of the FastAPI app. Has extra deps (`minio`, `pika`) not in `requirements.txt`. Tests do not cover this file.

## Developer commands

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Run API server
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Run all tests (no running Ollama needed — tests mock httpx)
.venv\Scripts\python.exe -m pytest

# Run a single test file
.venv\Scripts\python.exe -m pytest tests\test_api.py -v

# Debug logging
$env:LOG_LEVEL = 'DEBUG'
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Build Ollama model (one-time)
ollama create evidencopilot -f Modelfile

# Demo script (requires running server)
python scripts\demo_all_endpoints.py

# Dev tunnel with ngrok
python scripts\start_ngrok_tunnel.py
```

## Setup gotchas

- `.env` is gitignored. Copy from `.env.example` and fill in.
- Ollama must be running locally with `evidencopilot:latest` for endpoints that call the model (`/process/claim`, `/review/paper` with `use_ai=true`).
- The matching endpoint (`/match/claim`) uses deterministic keyword scoring — no Ollama needed.
- README paths reference `E:\Code\SEP490\...` but actual root is `D:\FPT\FA26\SEP490\Prototype_2\ollama-claim-api`.
- MinerU is optional for PDF/DOCX parsing; falls back to PyMuPDF / python-docx.

## Testing quirks

- Tests mock Ollama calls (httpx.AsyncClient monkeypatch), so no running Ollama required.
- `test_demo_backend.py` clears `demo_store` between tests via `reset_state` autouse fixture.
- Tests use `app.dependency_overrides` for `get_settings` — always override settings in test setup.
- No linter or typechecker configured.

## Key architecture

| Directory/File | Purpose |
|---|---|
| `app/main.py` | FastAPI entrypoint, routers, exception handlers |
| `app/store.py` | `DemoStore` — in-memory dicts for sources & papers |
| `app/ollama_client.py` | httpx calls to Ollama `/api/generate`, `/api/embeddings`, `/api/tags` |
| `app/extraction.py` | Text extraction from txt/md/docx/pdf, optional MinerU |
| `app/matching.py` | Keyword TF overlap scoring (no AI) |
| `app/paper_review.py` | Rule-based section detection + style comparison |
| `app/references.py` | Regex-based reference extraction |
| `app/models.py` | Pydantic request/response models |
| `app/settings.py` | Settings loaded from env with `@lru_cache` |
| `app/env.py` | Thin dotenv loader |
| `app/logging_config.py` | LOG_LEVEL-based StreamHandler |
| `scripts/` | `start_ngrok_tunnel.py`, `demo_all_endpoints.py` |
| `ai_worker.py` | Production worker (MinIO + RabbitMQ) — separate from FastAPI |
