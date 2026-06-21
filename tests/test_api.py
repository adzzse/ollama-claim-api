import pytest
from fastapi.testclient import TestClient

from app.main import app, get_settings, run_claim_analysis
from app.models import ClaimAnalysisRequest, ClaimAnalysisResponse
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    analyze_claim,
    analyze_paper_review,
)
from app.settings import Settings
from app.store import PaperRecord, PaperSection


API_KEY = "test-key"


def override_settings() -> Settings:
    return Settings()


@pytest.fixture(autouse=True)
def reset_overrides():
    app.dependency_overrides[get_settings] = override_settings
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def sample_payload() -> dict:
    return {
        "claim": "Traceable evidence improves review quality.",
        "source_id": "source-1",
        "title": "agile-risk-management.pdf",
        "excerpt": "Evidence traceability links claims to source material so reviewers can evaluate support quality.",
    }


def auth_headers() -> dict:
    return {"X-API-Key": API_KEY}


def test_process_claim_accepts_missing_api_key(client: TestClient):
    async def fake_analysis():
        return ClaimAnalysisResponse(
            verdict="supported",
            confidence=0.92,
            matched_source_ids=["source-1"],
            missing_evidence=[],
            explanation="The provided source directly supports the claim.",
        )

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", json=sample_payload())

    assert response.status_code == 200
    assert response.json()["verdict"] == "supported"


def test_process_claim_rejects_empty_claim(client: TestClient):
    payload = sample_payload()
    payload["claim"] = "   "

    response = client.post("/process/claim", headers=auth_headers(), json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_empty_source_id(client: TestClient):
    payload = sample_payload()
    payload["source_id"] = "   "

    response = client.post("/process/claim", headers=auth_headers(), json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_missing_excerpt(client: TestClient):
    payload = sample_payload()
    del payload["excerpt"]

    response = client.post("/process/claim", headers=auth_headers(), json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_oversized_excerpt(client: TestClient):
    payload = sample_payload()
    payload["excerpt"] = "x" * 4001

    response = client.post("/process/claim", headers=auth_headers(), json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_old_sources_shape(client: TestClient):
    payload = {
        "claim": "Traceable evidence improves review quality.",
        "sources": [
            {
                "id": "source-1",
                "title": "agile-risk-management.pdf",
                "excerpt": "Evidence traceability links claims to source material.",
            }
        ],
    }

    response = client.post("/process/claim", headers=auth_headers(), json=payload)

    assert response.status_code == 422


def test_process_claim_returns_structured_model_result(client: TestClient):
    async def fake_analysis():
        return ClaimAnalysisResponse(
            verdict="supported",
            confidence=0.92,
            matched_source_ids=["source-1"],
            missing_evidence=[],
            explanation="The provided source directly supports the claim.",
        )

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", headers=auth_headers(), json=sample_payload())

    assert response.status_code == 200
    assert response.json() == {
        "verdict": "supported",
        "confidence": 0.92,
        "matched_source_ids": ["source-1"],
        "missing_evidence": [],
        "explanation": "The provided source directly supports the claim.",
    }


def test_process_claim_returns_503_when_ollama_is_unavailable(client: TestClient):
    async def fake_analysis():
        raise OllamaUnavailableError("Ollama generation failed")

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", headers=auth_headers(), json=sample_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama generation failed"


def test_process_claim_returns_502_when_model_json_is_invalid(client: TestClient):
    async def fake_analysis():
        raise OllamaInvalidResponseError("Ollama returned invalid claim-analysis JSON")

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", headers=auth_headers(), json=sample_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "Ollama returned invalid claim-analysis JSON"


def test_health_returns_ollama_status(client: TestClient):
    async def fake_health():
        return {"ok": True, "model_available": True}

    from app.main import check_ollama_health

    app.dependency_overrides[check_ollama_health] = fake_health

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "evidencopilot:latest",
        "ollama": {"ok": True, "model_available": True},
    }


def test_analyze_claim_logs_ai_verdict_and_explanation(monkeypatch, caplog):
    import asyncio
    import json
    import logging

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": json.dumps(
                    {
                        "verdict": "supported",
                        "confidence": 0.87,
                        "matched_source_ids": ["source-1"],
                        "missing_evidence": [],
                        "reasoning_summary": "I compared the claim terms against source-1 and found direct support.",
                        "explanation": "The source directly links traceability to review quality.",
                    }
                )
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            assert json["think"] is False
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)
    caplog.set_level(logging.DEBUG, logger="app")

    result = asyncio.run(
        analyze_claim(
            ClaimAnalysisRequest.model_validate(sample_payload()),
            Settings(),
        )
    )

    assert result.verdict == "supported"
    messages = [record.getMessage() for record in caplog.records]
    assert any("ai verdict verdict=supported confidence=0.87" in message for message in messages)
    assert any(
        "ai reasoning summary I compared the claim terms against source-1 and found direct support." in message
        for message in messages
    )
    assert any(
        "ai explanation The source directly links traceability to review quality." in message
        for message in messages
    )


def test_analyze_paper_review_uses_configured_timeout(monkeypatch):
    import asyncio
    import json

    captured_timeouts = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": json.dumps(
                    {
                        "paper_id": "paper-1",
                        "detected_style": "conference",
                        "target_style": "conference",
                        "missing_sections": [],
                        "weak_sections": [],
                        "claim_recommendations": [],
                        "reasoning_summary": "I checked the requested style against the provided sections.",
                    }
                )
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            captured_timeouts.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)
    record = PaperRecord(
        id="paper-1",
        filename="paper.txt",
        text="Abstract\nTraceability study.",
        sections=[PaperSection(name="Abstract", text="Traceability study.")],
    )

    asyncio.run(
        analyze_paper_review(
            record,
            "conference",
            {
                "paper_id": "paper-1",
                "detected_style": "conference",
                "target_style": "conference",
                "missing_sections": [],
                "weak_sections": [],
                "claim_recommendations": [],
            },
            Settings(ollama_paper_timeout_seconds=321.0),
        )
    )

    assert captured_timeouts == [321.0]


def test_analyze_paper_review_logs_timeout_exception_type(monkeypatch, caplog):
    import asyncio
    import logging

    import httpx

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            raise httpx.ReadTimeout("")

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)
    caplog.set_level(logging.INFO, logger="app.ollama_client")
    record = PaperRecord(
        id="paper-1",
        filename="paper.txt",
        text="Abstract\nTraceability study.",
        sections=[PaperSection(name="Abstract", text="Traceability study.")],
    )

    with pytest.raises(OllamaUnavailableError):
        asyncio.run(
            analyze_paper_review(
                record,
                "conference",
                {
                    "paper_id": "paper-1",
                    "detected_style": "conference",
                    "target_style": "conference",
                    "missing_sections": [],
                    "weak_sections": [],
                    "claim_recommendations": [],
                },
                Settings(ollama_paper_timeout_seconds=321.0),
            )
        )

    assert "paper ai request failed model=evidencopilot:latest exception_type=ReadTimeout" in caplog.text
    assert "timeout_seconds=321.0" in caplog.text


def test_get_embeddings_success(client: TestClient):
    from app.models import EmbeddingResponse

    async def fake_run_generate_embeddings():
        return EmbeddingResponse(embedding=[0.1, -0.2, 0.35])

    from app.main import run_generate_embeddings
    app.dependency_overrides[run_generate_embeddings] = fake_run_generate_embeddings

    response = client.post("/ai/embeddings", json={"text": "hello world"})

    assert response.status_code == 200
    assert response.json() == {"embedding": [0.1, -0.2, 0.35]}


def test_get_embeddings_validation_error_empty_text(client: TestClient):
    response = client.post("/ai/embeddings", json={"text": "   "})
    assert response.status_code == 422


def test_get_embeddings_validation_error_missing_field(client: TestClient):
    response = client.post("/ai/embeddings", json={})
    assert response.status_code == 422


def test_generate_embeddings_client_success(monkeypatch):
    import asyncio
    from app.ollama_client import generate_embeddings

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": [0.01, -0.02, 0.03]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            assert url.endswith("/api/embeddings")
            assert json["prompt"] == "test text"
            assert json["model"] == "nomic-embed-text"
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    res = asyncio.run(generate_embeddings("test text", Settings()))
    assert res == [0.01, -0.02, 0.03]


def test_generate_embeddings_client_fallback_embeddings(monkeypatch):
    import asyncio
    from app.ollama_client import generate_embeddings

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.99, -0.88]]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    res = asyncio.run(generate_embeddings("test text", Settings()))
    assert res == [0.99, -0.88]


def test_generate_embeddings_client_unavailable(monkeypatch):
    import asyncio
    import httpx
    from app.ollama_client import generate_embeddings

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    with pytest.raises(OllamaUnavailableError):
        asyncio.run(generate_embeddings("test text", Settings()))


def test_generate_embeddings_client_invalid_response(monkeypatch):
    import asyncio
    from app.ollama_client import generate_embeddings

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"invalid_key": "not a list"}

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    with pytest.raises(OllamaInvalidResponseError):
        asyncio.run(generate_embeddings("test text", Settings()))
