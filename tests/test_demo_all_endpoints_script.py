from pathlib import Path

from scripts.demo_all_endpoints import DemoConfig, run_demo


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def get(self, path: str):
        self.calls.append(("GET", path))
        if path == "/health":
            return FakeResponse({"status": "ok"})
        if path == "/sources":
            return FakeResponse({"sources": [{"id": "source-1", "filename": "source.txt", "chunk_count": 1}]})
        if path == "/sources/references":
            return FakeResponse({"references": []})
        raise AssertionError(f"Unexpected GET {path}")

    def post(self, path: str, **kwargs):
        self.calls.append(("POST", path))
        if path == "/sources":
            assert kwargs["files"]["files"][0] == "demo-source.txt"
            return FakeResponse({"sources": [{"id": "source-1", "filename": "demo-source.txt", "chunk_count": 1}]})
        if path == "/match/claim":
            assert kwargs["json"]["claim"]
            return FakeResponse({"claim": kwargs["json"]["claim"], "matches": [{"source_id": "source-1"}]})
        if path == "/papers":
            assert kwargs["files"]["file"][0] == "demo-paper.txt"
            return FakeResponse({"paper": {"id": "paper-1", "filename": "demo-paper.txt", "section_count": 3}})
        if path == "/review/paper":
            assert kwargs["json"]["paper_id"] == "paper-1"
            assert kwargs["json"]["use_ai"] is True
            return FakeResponse({"paper_id": "paper-1", "missing_sections": []})
        if path == "/process/claim":
            assert kwargs["json"]["source_id"] == "source-1"
            assert kwargs["json"]["excerpt"]
            return FakeResponse({"verdict": "partially_supported"})
        raise AssertionError(f"Unexpected POST {path}")


def test_demo_runner_calls_every_public_endpoint(tmp_path: Path):
    client = FakeClient()
    config = DemoConfig(
        base_url="http://127.0.0.1:8000",
        review_paper_use_ai=True,
        work_dir=tmp_path,
    )

    results = run_demo(
        config,
        client=client,
    )

    assert client.calls == [
        ("GET", "/health"),
        ("POST", "/sources"),
        ("GET", "/sources"),
        ("GET", "/sources/references"),
        ("POST", "/match/claim"),
        ("POST", "/papers"),
        ("POST", "/review/paper"),
        ("POST", "/process/claim"),
    ]
    assert set(results) == {
        "health",
        "upload_sources",
        "list_sources",
        "source_references",
        "match_claim",
        "upload_paper",
        "review_paper",
        "process_claim",
    }
