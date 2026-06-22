# EvidencePilot Python AI Worker

Stateless FastAPI worker for the Java backend.

Java owns uploads, persistence, source/chunk/reference records, graph JSON, auth, projects, datasets, and orchestration. This service only runs AI worker functions:

- MinerU document extraction
- Ollama text generation and claim analysis
- Ollama embeddings

The Java backend calls this service with `AI_MODEL_BASE_URL`, for example `AI_MODEL_BASE_URL=http://host.docker.internal:8000`.

## Setup

```powershell
cd E:\Code\SEP490\ollama-claim-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Build the project Ollama model once:

```powershell
ollama create evidencopilot -f Modelfile
```

## Run

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For detailed logs:

```powershell
$env:LOG_LEVEL = 'DEBUG'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## MinerU

For document extraction, install MinerU separately and point `MINERU_COMMAND` at the executable if needed:

```text
MINERU_COMMAND=E:\Code\SEP490\.venv-mineru\Scripts\mineru.exe
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_TIMEOUT_SECONDS=600
```

`MINERU_BACKEND=pipeline` is useful on this local CPU-only MinerU setup.

## Endpoints

`GET /health`

Returns service and Ollama health.

`POST /extract`

Multipart form field: `file`

Returns extracted Markdown only. It does not create source records or persist upload state.

```json
{
  "filename": "original.pdf",
  "method": "mineru",
  "markdown": "# Extracted document\n\n..."
}
```

PowerShell example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/extract `
  -F "file=@E:\Code\SEP490\sources\example.pdf"
```

`POST /ai/embeddings`

```json
{
  "text": "Evidence traceability links claims to source material."
}
```

Response:

```json
{
  "embedding": [0.1, -0.2, 0.3]
}
```

`POST /ai/generate`

```json
{
  "prompt": "Explain evidence traceability."
}
```

Response:

```json
{
  "model": "evidencopilot:latest",
  "response": "Generated text...",
  "done": true
}
```

`POST /process/claim`

Kept for Java flows that still delegate claim verdict/explanation generation.

```json
{
  "claim": "Traceable evidence improves review quality.",
  "source_id": "source-1",
  "title": "agile-risk-management.pdf",
  "excerpt": "Evidence traceability links claims to source material..."
}
```

Response:

```json
{
  "verdict": "supported",
  "confidence": 0.92,
  "matched_source_ids": ["source-1"],
  "missing_evidence": [],
  "explanation": "The provided source directly supports the claim."
}
```

## Ngrok

Start the local API and expose it through ngrok:

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\Activate.ps1
python scripts\start_ngrok_tunnel.py
```

If the backend is already running in another terminal, start only the tunnel:

```powershell
python scripts\start_ngrok_tunnel.py --no-server
```

## Tests

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\python.exe -m pytest -q
```
