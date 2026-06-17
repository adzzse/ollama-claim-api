# Ollama Claim Analysis API

Standalone FastAPI service for demoing EvidencePilot AI flows with local Ollama/Qwen.

It supports two demo workflows:

- Claim evidence matching: upload source files, then match a user claim against extracted source chunks.
- Paper structure review: upload a user paper, then review missing or weak sections for a target style.

The demo backend stores uploaded sources and papers in memory. Restarting `uvicorn` clears them.

## Setup

```powershell
cd E:\Code\SEP490\ollama-claim-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

The local demo API does not require an API key.

Optional ngrok setup for exposing the local API:

```powershell
winget install Ngrok.Ngrok
ngrok update
ngrok config add-authtoken <your ngrok authtoken>
```

If `ngrok` is installed by WinGet but is not on `PATH`, use the generated shim path when starting the tunnel:

```text
C:\Users\HoangAnhDo\AppData\Local\Microsoft\WinGet\Links\ngrok.exe
```

## Run

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

To see detailed processing logs for upload, extraction, chunking, reference extraction, matching, and paper review:

```powershell
$env:LOG_LEVEL = 'DEBUG'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

When MinerU runs, DEBUG logs also stream MinerU `stdout` and `stderr` lines with `mineru stdout ...` and `mineru stderr ...` prefixes.

For `/process/claim`, the terminal logs the AI request summary, parsed verdict, confidence, matched source IDs, missing evidence, and returned explanation. With `LOG_LEVEL=DEBUG`, it also logs the full prompt, raw model JSON response, and the model-provided `reasoning_summary`.

## Run With Ngrok

Start the local API and expose it through ngrok:

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\Activate.ps1
python scripts\start_ngrok_tunnel.py
```

The script starts `uvicorn` on `http://127.0.0.1:8000`, then starts `ngrok http http://127.0.0.1:8000`. Copy the public `Forwarding` HTTPS URL from the ngrok output.

Before starting the local API, the script runs `ngrok diagnose`. If ngrok cannot establish TLS to its tunnel servers, the script exits before starting `uvicorn` so it does not sit in a reconnect loop.

If you see `failed to fetch CRL` or `ERR_NGROK_8008`, the local API is not the failing layer. Check VPN/proxy/firewall TLS inspection, system time, and whether the network allows ngrok's TLS and certificate revocation-list checks:

```powershell
ngrok diagnose
```

If PowerShell cannot find `ngrok`, pass the full executable path:

```powershell
python scripts\start_ngrok_tunnel.py `
  --ngrok-path C:\Users\HoangAnhDo\AppData\Local\Microsoft\WinGet\Links\ngrok.exe
```

If the backend is already running in another terminal, start only the tunnel:

```powershell
python scripts\start_ngrok_tunnel.py --no-server
```

Run the demo script against the ngrok URL:

```powershell
python scripts\demo_all_endpoints.py `
  --base-url https://your-forwarding-url.ngrok-free.dev
```

Build the project-specific Ollama model once:

```powershell
ollama create evidencopilot -f Modelfile
```

Keep Ollama running locally with the model:

```powershell
ollama list
```

Expected model:

```text
evidencopilot:latest
```

## Local Requests

Health:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

Analyze a claim:

```powershell
$body = @{
  claim = 'Traceable evidence improves review quality.'
  source_id = 'source-1'
  title = 'agile-risk-management.pdf'
  excerpt = 'Evidence traceability links claims to source material so reviewers can evaluate support quality.'
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/process/claim `
  -ContentType 'application/json' `
  -Body $body
```

## Demo Flow: Upload Sources And Match A Claim

Upload one or more `.txt`, `.md`, `.docx`, or clean text-based `.pdf` sources:

```powershell
curl.exe -X POST http://127.0.0.1:8000/sources `
  -F "files=@E:\Code\SEP490\ollama-claim-api\sources\example-source.txt"
```

For `.pdf` and `.docx` uploads, the backend tries MinerU first and uses the generated Markdown as the extracted text. If the `mineru` command is not installed or the MinerU run fails, the backend falls back to the built-in PyMuPDF / `python-docx` extractors.

Install MinerU separately when you want higher-fidelity parsing:

```powershell
.\.venv\Scripts\Activate.ps1
pip install "mineru[core]"
```

Optional MinerU settings in `.env`:

```text
MINERU_COMMAND=mineru
MINERU_METHOD=auto
MINERU_TIMEOUT_SECONDS=600
LOG_LEVEL=INFO
```

If MinerU is installed in a separate virtual environment, point `MINERU_COMMAND` at the full executable path, for example:

```text
MINERU_COMMAND=E:\Code\SEP490\.venv-mineru\Scripts\mineru.exe
```

List ingested sources:

```powershell
Invoke-RestMethod -Method Get `
  -Uri http://127.0.0.1:8000/sources
```

Export extracted references as JSON for graph features:

```powershell
Invoke-RestMethod -Method Get `
  -Uri http://127.0.0.1:8000/sources/references
```

Response shape:

```json
{
  "references": [
    {
      "id": "source-1-ref-1",
      "source_id": "source-1",
      "filename": "example-source.txt",
      "raw_text": "Smith, J. (2024). Evidence Traceability for Review Workflows. Journal of Review Systems.",
      "title": "Evidence Traceability for Review Workflows",
      "year": 2024
    }
  ]
}
```

Match a claim against stored source chunks:

```powershell
$body = @{
  claim = 'Traceable evidence improves review quality.'
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/match/claim `
  -ContentType 'application/json' `
  -Body $body
```

Response shape:

```json
{
  "claim": "Traceable evidence improves review quality.",
  "matches": [
    {
      "source_id": "source-1",
      "filename": "example-source.pdf",
      "chunk_id": "source-1-chunk-1",
      "page": null,
      "excerpt": "Evidence traceability links claims to source material...",
      "score": 0.62,
      "suitability": "strong",
      "explanation": "This source chunk shares strong terminology with the claim and is a good candidate for review."
    }
  ]
}
```

This matching endpoint uses deterministic keyword scoring for demo reliability. The `/process/claim` endpoint calls Qwen/Ollama for final claim support judgment when you already have one selected excerpt.

## Demo Flow: Upload And Review A Paper

Upload a user paper:

```powershell
curl.exe -X POST http://127.0.0.1:8000/papers `
  -F "file=@E:\Code\SEP490\sources\student-paper.docx"
```

Review it against a target style:

```powershell
$body = @{
  paper_id = 'paper-1'
  target_style = 'conference'
  use_ai = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/review/paper `
  -ContentType 'application/json' `
  -Body $body
```

If `target_style` is omitted, the backend uses the auto-detected style. `use_ai` defaults to `false` for the fast local rule scan; set it to `true` to call Ollama for the paper review. Supported target styles:

```text
conference, article, magazine, report, thesis
```

Response shape:

```json
{
  "paper_id": "paper-1",
  "detected_style": "conference",
  "target_style": "conference",
  "missing_sections": [
    {
      "section": "Methodology",
      "issue": "Methodology is expected for a conference paper but was not found.",
      "recommendation": "Add a Methodology section or rename the matching content clearly."
    }
  ],
  "weak_sections": [],
  "claim_recommendations": []
}
```

## Cloudflare Tunnel

Forward the public HTTPS hostname to the local service:

```text
http://127.0.0.1:8000
```

Protect the hostname with Cloudflare Access service tokens before exposing this local demo outside your machine.

Remote requests should include:

```text
CF-Access-Client-Id: <cloudflare service token id>
CF-Access-Client-Secret: <cloudflare service token secret>
Content-Type: application/json
```

## API Contract

`POST /process/claim`

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

Allowed verdicts: `supported`, `partially_supported`, `unsupported`, `unclear`.

## Tests

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\python.exe -m pytest
```

## Ready Demo Test For All Endpoints

Start the backend first:

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, run the all-endpoint demo test:

```powershell
cd E:\Code\SEP490\ollama-claim-api
.\.venv\Scripts\Activate.ps1
python scripts\demo_all_endpoints.py
```

To make the paper-review demo call Ollama, add `--review-paper-ai`:

```powershell
python scripts\demo_all_endpoints.py --review-paper-ai
```

The local demo does not require `--api-key`; the flag remains only for compatibility with an external proxy that expects one.

The script automatically creates demo `.txt` files and tests:

```text
GET  /health
POST /sources
GET  /sources
GET  /sources/references
POST /match/claim
POST /papers
POST /review/paper
POST /process/claim
```

For Cloudflare Tunnel, pass the public URL and keep your Cloudflare Access token values in environment variables:

```powershell
$env:CF_ACCESS_CLIENT_ID = '<cloudflare service token id>'
$env:CF_ACCESS_CLIENT_SECRET = '<cloudflare service token secret>'

python scripts\demo_all_endpoints.py `
  --base-url https://api.yourdomain.com
```

The script prints the JSON response from every endpoint. If any endpoint fails, it stops with the HTTP error.
