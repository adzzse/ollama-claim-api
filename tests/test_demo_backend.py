import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_settings
from app.settings import Settings


API_KEY = "test-key"


def override_settings() -> Settings:
    return Settings()


@pytest.fixture(autouse=True)
def reset_state():
    from app.store import demo_store

    demo_store.clear()
    app.dependency_overrides[get_settings] = override_settings
    yield
    app.dependency_overrides.clear()
    demo_store.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def auth_headers() -> dict:
    return {"X-API-Key": API_KEY}


def test_upload_text_source_extracts_chunks_and_lists_source(client: TestClient):
    response = client.post(
        "/sources",
        headers=auth_headers(),
        files={
            "files": (
                "traceability.txt",
                b"Evidence traceability links claims to source material.\nReviewers use mapped evidence to judge support quality.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["filename"] == "traceability.txt"
    assert body["sources"][0]["chunk_count"] >= 1

    list_response = client.get("/sources", headers=auth_headers())
    assert list_response.status_code == 200
    assert list_response.json()["sources"][0]["filename"] == "traceability.txt"


def test_upload_and_match_emit_processing_debug_logs(client: TestClient, caplog: pytest.LogCaptureFixture):
    import logging

    caplog.set_level(logging.DEBUG, logger="app")

    upload_response = client.post(
        "/sources",
        headers=auth_headers(),
        files={
            "files": (
                "traceability.txt",
                b"Evidence traceability links claims to source material.\n"
                b"Reviewers use mapped evidence to judge support quality.",
                "text/plain",
            )
        },
    )
    match_response = client.post(
        "/match/claim",
        headers=auth_headers(),
        json={"claim": "Traceable evidence improves review quality."},
    )

    assert upload_response.status_code == 200
    assert match_response.status_code == 200

    messages = [record.getMessage() for record in caplog.records]
    assert any("upload source start" in message for message in messages)
    assert any("extract upload text" in message for message in messages)
    assert any("stored source" in message for message in messages)
    assert any("match claim complete" in message for message in messages)


def test_upload_source_extracts_references_for_graph_json(client: TestClient):
    source_text = (
        "Evidence Review\n"
        "Traceability improves evidence review quality.\n\n"
        "References\n"
        "[1] Smith, J. (2024). Evidence Traceability for Review Workflows. Journal of Review Systems.\n"
        "[2] Nguyen, A. (2023). Source Graphs for Academic Claims. Conference on Learning Tools."
    )

    upload_response = client.post(
        "/sources",
        files={"files": ("references.txt", source_text.encode("utf-8"), "text/plain")},
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["sources"][0]["reference_count"] == 2

    response = client.get("/sources/references")

    assert response.status_code == 200
    body = response.json()
    assert body["references"] == [
        {
            "id": "source-1-ref-1",
            "source_id": "source-1",
            "filename": "references.txt",
            "raw_text": "Smith, J. (2024). Evidence Traceability for Review Workflows. Journal of Review Systems.",
            "title": "Evidence Traceability for Review Workflows",
            "year": 2024,
        },
        {
            "id": "source-1-ref-2",
            "source_id": "source-1",
            "filename": "references.txt",
            "raw_text": "Nguyen, A. (2023). Source Graphs for Academic Claims. Conference on Learning Tools.",
            "title": "Source Graphs for Academic Claims",
            "year": 2023,
        },
    ]


def test_upload_source_extracts_mineru_markdown_references_for_graph_json(client: TestClient):
    source_text = (
        "## Acknowledgements\n\n"
        "This work was supported by the project team.\n\n"
        "## References\n\n"
        "Arvidson, T., Gasch, J., Goward, S.N., 2001. Landsat 7's long-term acquisition plan. "
        "Remote Sens. Environ. 78: 13-26. http://dx.doi.org/10.1016/S0034-4257(01)00263-2.\n\n"
        "Baumann, P., Mazzetti, P., Ungar, J., Barbera, R., Barboni, D., Beccati, A., Bigagli, L., "
        "Boldrini, E., Bruno, R., Calanducci, A., Campalani, P., Clements, O., Dumitru, A., Grant, M., "
        "Herzig, P., Kakaletris, G., Laxton, J., Koltsida, P., Lipskoch, K., Mahdiraji, A.R., Mantovani, S., "
        "Merticariu, V., Messina, A., Misev, D., Natali, S., Nativi, S., Oosthoek, J., Pappalardo, M., "
        "Passmore, J., Rossi, A.P., Rundo, F., Sen, M., Sorbera, V., Sullivan, D., Torrisi, M., "
        "Trovato, L., Veratelli, M.G., Wagner, S., 2016. Big data analytics for earth sciences: "
        "the EarthServer approach. Int. J. Digital Earth 9:3-29.\n\n"
        "Australian Academy of Science, 2009. An Australian Strategic Plan for Earth Observations from Space. "
        "Australian Academy of Science, Canberra.\n\n"
        "## Appendix\n\n"
        "Not a reference."
    )

    upload_response = client.post(
        "/sources",
        files={"files": ("mineru.md", source_text.encode("utf-8"), "text/markdown")},
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["sources"][0]["reference_count"] == 3

    response = client.get("/sources/references")

    assert response.status_code == 200
    body = response.json()
    assert [reference["year"] for reference in body["references"]] == [2001, 2016, 2009]
    assert body["references"][0]["title"] == "Landsat 7's long-term acquisition plan"
    assert body["references"][1]["title"] == "Big data analytics for earth sciences: the EarthServer approach"
    assert body["references"][2]["title"] == "An Australian Strategic Plan for Earth Observations from Space"


def test_upload_pdf_source_prefers_mineru_extraction(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_extract_with_mineru(filename: str, raw: bytes) -> str:
        assert filename == "traceability.pdf"
        assert raw == b"not-a-real-pdf"
        return "MinerU extracted traceability text for reviewers."

    monkeypatch.setattr("app.extraction.extract_with_mineru", fake_extract_with_mineru)

    response = client.post(
        "/sources",
        headers=auth_headers(),
        files={"files": ("traceability.pdf", b"not-a-real-pdf", "application/pdf")},
    )

    assert response.status_code == 200

    match_response = client.post(
        "/match/claim",
        headers=auth_headers(),
        json={"claim": "MinerU extracts traceability text."},
    )
    assert match_response.status_code == 200
    assert "MinerU extracted traceability" in match_response.json()["matches"][0]["excerpt"]


def test_mineru_extraction_loads_command_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from io import StringIO
    import subprocess

    from app.extraction import extract_with_mineru

    expected_command = r"C:\Tools\MinerU\mineru.exe"
    tmp_path.joinpath(".env").write_text(
        f"MINERU_COMMAND={expected_command}\nMINERU_METHOD=ocr\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINERU_COMMAND", raising=False)
    monkeypatch.delenv("MINERU_METHOD", raising=False)

    def fake_which(command: str) -> str | None:
        return command if command == expected_command else None

    class FakeProcess:
        stdout = StringIO("")
        stderr = StringIO("")

        def __init__(self, args, **kwargs):
            assert args[0] == expected_command
            assert args[args.index("--method") + 1] == "ocr"
            actual_output_dir = args[args.index("--output") + 1]
            type(tmp_path)(actual_output_dir).joinpath("result.md").write_text(
                "MinerU extracted text.",
                encoding="utf-8",
            )

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("app.extraction.shutil.which", fake_which)
    monkeypatch.setattr("app.extraction.subprocess.Popen", FakeProcess)

    assert extract_with_mineru("paper.pdf", b"%PDF") == "MinerU extracted text."


def test_mineru_extraction_streams_process_output_to_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
):
    from io import StringIO
    import logging

    from app.extraction import extract_with_mineru

    expected_command = r"C:\Tools\MinerU\mineru.exe"
    monkeypatch.setenv("MINERU_COMMAND", expected_command)
    monkeypatch.setenv("MINERU_METHOD", "auto")
    caplog.set_level(logging.DEBUG, logger="app")

    def fake_which(command: str) -> str | None:
        return command if command == expected_command else None

    class FakeProcess:
        stdout = StringIO("loading model\nparsing page 1\n")
        stderr = StringIO("warning: slow page\n")

        def __init__(self, args, **kwargs):
            actual_output_dir = args[args.index("--output") + 1]
            type(tmp_path)(actual_output_dir).joinpath("result.md").write_text(
                "MinerU extracted text.",
                encoding="utf-8",
            )

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("app.extraction.shutil.which", fake_which)
    monkeypatch.setattr("app.extraction.subprocess.Popen", FakeProcess)

    assert extract_with_mineru("paper.pdf", b"%PDF") == "MinerU extracted text."

    messages = [record.getMessage() for record in caplog.records]
    assert any("mineru stdout loading model" in message for message in messages)
    assert any("mineru stdout parsing page 1" in message for message in messages)
    assert any("mineru stderr warning: slow page" in message for message in messages)


def test_upload_pdf_source_falls_back_when_mineru_is_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from app.extraction import MinerUUnavailableError

    def fake_extract_with_mineru(filename: str, raw: bytes) -> str:
        raise MinerUUnavailableError("MinerU command is not available")

    def fake_extract_pdf(raw: bytes) -> str:
        return "Fallback PDF extraction still works."

    monkeypatch.setattr("app.extraction.extract_with_mineru", fake_extract_with_mineru)
    monkeypatch.setattr("app.extraction._extract_pdf", fake_extract_pdf)

    response = client.post(
        "/sources",
        headers=auth_headers(),
        files={"files": ("fallback.pdf", b"pdf bytes", "application/pdf")},
    )

    assert response.status_code == 200

    match_response = client.post(
        "/match/claim",
        headers=auth_headers(),
        json={"claim": "Fallback PDF extraction works."},
    )
    assert match_response.status_code == 200
    assert "Fallback PDF extraction" in match_response.json()["matches"][0]["excerpt"]


def test_match_claim_returns_suitable_source_chunks(client: TestClient):
    client.post(
        "/sources",
        headers=auth_headers(),
        files={
            "files": (
                "traceability.txt",
                b"Evidence traceability links claims to source material so reviewers can evaluate support quality.\n"
                b"Unrelated project scheduling notes describe sprint dates.",
                "text/plain",
            )
        },
    )

    response = client.post(
        "/match/claim",
        headers=auth_headers(),
        json={"claim": "Traceable evidence improves review quality."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["claim"] == "Traceable evidence improves review quality."
    assert body["matches"]
    assert body["matches"][0]["filename"] == "traceability.txt"
    assert "traceability" in body["matches"][0]["excerpt"].lower()
    assert body["matches"][0]["suitability"] in {"strong", "medium", "weak"}


def test_match_claim_requires_ingested_sources(client: TestClient):
    response = client.post(
        "/match/claim",
        headers=auth_headers(),
        json={"claim": "Traceable evidence improves review quality."},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "No sources have been ingested"


def test_upload_paper_and_review_missing_conference_sections(client: TestClient):
    paper_text = (
        "Abstract\n"
        "This paper studies traceability.\n\n"
        "Introduction\n"
        "Traceability helps reviewers inspect claims.\n\n"
        "Conclusion\n"
        "The paper summarizes the benefits."
    )
    upload_response = client.post(
        "/papers",
        headers=auth_headers(),
        files={"file": ("paper.txt", paper_text.encode("utf-8"), "text/plain")},
    )
    assert upload_response.status_code == 200
    paper_id = upload_response.json()["paper"]["id"]

    response = client.post(
        "/review/paper",
        headers=auth_headers(),
        json={"paper_id": paper_id, "target_style": "conference"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_style"] == "conference"
    assert body["detected_style"] in {"conference", "article", "report", "magazine", "unknown"}
    missing_names = {item["section"] for item in body["missing_sections"]}
    assert "Methodology" in missing_names
    assert "Results" in missing_names
    assert body["weak_sections"]
    assert body["claim_recommendations"]


def test_review_paper_uses_ai_when_requested(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    paper_text = (
        "Abstract\n"
        "This paper studies traceability.\n\n"
        "Introduction\n"
        "Traceability helps reviewers inspect claims."
    )
    upload_response = client.post(
        "/papers",
        headers=auth_headers(),
        files={"file": ("paper.txt", paper_text.encode("utf-8"), "text/plain")},
    )
    paper_id = upload_response.json()["paper"]["id"]
    calls = []

    async def fake_ai_review(record, target_style, baseline_review, settings):
        calls.append((record.id, target_style, baseline_review["target_style"], settings.ollama_model))
        return {
            "paper_id": record.id,
            "detected_style": "conference",
            "target_style": "conference",
            "missing_sections": [
                {
                    "section": "Results",
                    "issue": "The AI review found no explicit results section.",
                    "recommendation": "Add a Results section with concrete findings.",
                }
            ],
            "weak_sections": [],
            "claim_recommendations": [],
        }

    monkeypatch.setattr("app.main.analyze_paper_review", fake_ai_review, raising=False)

    response = client.post(
        "/review/paper",
        headers=auth_headers(),
        json={"paper_id": paper_id, "target_style": "conference", "use_ai": True},
    )

    assert response.status_code == 200
    assert calls == [(paper_id, "conference", "conference", "evidencopilot:latest")]
    assert response.json()["missing_sections"][0]["issue"] == "The AI review found no explicit results section."


def test_review_paper_logs_when_ai_review_is_skipped(client: TestClient, caplog: pytest.LogCaptureFixture):
    paper_text = (
        "Abstract\n"
        "This paper studies traceability.\n\n"
        "Introduction\n"
        "Traceability helps reviewers inspect claims."
    )
    upload_response = client.post(
        "/papers",
        headers=auth_headers(),
        files={"file": ("paper.txt", paper_text.encode("utf-8"), "text/plain")},
    )
    paper_id = upload_response.json()["paper"]["id"]

    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.post(
            "/review/paper",
            headers=auth_headers(),
            json={"paper_id": paper_id, "target_style": "conference"},
        )

    assert response.status_code == 200
    assert f"review paper ai skipped paper_id={paper_id} use_ai=false" in caplog.text


def test_review_paper_uses_auto_detected_style_when_target_missing(client: TestClient):
    paper_text = (
        "Lead\n"
        "A short hook introduces the topic.\n\n"
        "Main Story\n"
        "The article explains traceability with examples.\n\n"
        "Takeaway\n"
        "Readers should map claims to evidence."
    )
    upload_response = client.post(
        "/papers",
        headers=auth_headers(),
        files={"file": ("magazine.txt", paper_text.encode("utf-8"), "text/plain")},
    )
    paper_id = upload_response.json()["paper"]["id"]

    response = client.post(
        "/review/paper",
        headers=auth_headers(),
        json={"paper_id": paper_id},
    )

    assert response.status_code == 200
    assert response.json()["target_style"] == response.json()["detected_style"]
