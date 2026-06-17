from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from dotenv import load_dotenv


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CLAIM = "Traceable evidence improves review quality."


class ResponseLike(Protocol):
    def raise_for_status(self) -> None:
        ...

    def json(self) -> dict[str, Any]:
        ...


class ClientLike(Protocol):
    def get(self, path: str) -> ResponseLike:
        ...

    def post(self, path: str, **kwargs: Any) -> ResponseLike:
        ...


@dataclass
class DemoConfig:
    base_url: str = DEFAULT_BASE_URL
    claim: str = DEFAULT_CLAIM
    target_style: str = "conference"
    review_paper_use_ai: bool = False
    work_dir: Path | None = None
    api_key: str | None = None


def run_demo(config: DemoConfig, client: ClientLike | None = None) -> dict[str, dict[str, Any]]:
    close_client = False
    if client is None:
        client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers=_headers(config.api_key),
            timeout=180.0,
        )
        close_client = True

    try:
        return _run_demo_with_client(config, client)
    finally:
        if close_client and hasattr(client, "close"):
            client.close()


def _run_demo_with_client(config: DemoConfig, client: ClientLike) -> dict[str, dict[str, Any]]:
    work_dir = config.work_dir or Path(tempfile.mkdtemp(prefix="ollama-api-demo-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    source_path, paper_path = _write_demo_files(work_dir)

    results: dict[str, dict[str, Any]] = {}
    results["health"] = _json(client.get("/health"))

    with source_path.open("rb") as source_file:
        results["upload_sources"] = _json(
            client.post(
                "/sources",
                files={"files": (source_path.name, source_file, "text/plain")},
            )
        )

    results["list_sources"] = _json(client.get("/sources"))
    results["source_references"] = _json(client.get("/sources/references"))

    results["match_claim"] = _json(
        client.post(
            "/match/claim",
            json={"claim": config.claim, "top_k": 5},
        )
    )

    with paper_path.open("rb") as paper_file:
        results["upload_paper"] = _json(
            client.post(
                "/papers",
                files={"file": (paper_path.name, paper_file, "text/plain")},
            )
        )

    paper_id = results["upload_paper"]["paper"]["id"]
    results["review_paper"] = _json(
        client.post(
            "/review/paper",
            json={
                "paper_id": paper_id,
                "target_style": config.target_style,
                "use_ai": config.review_paper_use_ai,
            },
        )
    )

    process_source = _source_for_process_claim(results)
    results["process_claim"] = _json(
        client.post(
            "/process/claim",
            json={"claim": config.claim, **process_source},
        )
    )

    return results


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    client_id = os.getenv("CF_ACCESS_CLIENT_ID")
    client_secret = os.getenv("CF_ACCESS_CLIENT_SECRET")
    if client_id and client_secret:
        headers["CF-Access-Client-Id"] = client_id
        headers["CF-Access-Client-Secret"] = client_secret
    return headers


def _json(response: ResponseLike) -> dict[str, Any]:
    response.raise_for_status()
    return response.json()


def _write_demo_files(work_dir: Path) -> tuple[Path, Path]:
    source_path = work_dir / "demo-source.txt"
    paper_path = work_dir / "demo-paper.txt"

    source_path.write_text(
        "Evidence traceability links claims to source material so reviewers can evaluate support quality.\n"
        "Mapped evidence helps reviewers inspect whether a claim is supported by the provided document.",
        encoding="utf-8",
    )
    paper_path.write_text(
        "Abstract\n"
        "This paper studies traceability for academic review.\n\n"
        "Introduction\n"
        "Traceability helps reviewers inspect claims in student writing.\n\n"
        "Conclusion\n"
        "The paper summarizes why mapped evidence is useful.",
        encoding="utf-8",
    )
    return source_path, paper_path


def _source_for_process_claim(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    matches = results.get("match_claim", {}).get("matches", [])
    if matches:
        first_match = matches[0]
        return {
            "source_id": first_match["source_id"],
            "title": first_match.get("filename", "matched-source"),
            "excerpt": first_match.get(
                "excerpt",
                "Evidence traceability links claims to source material so reviewers can evaluate support quality.",
            ),
        }

    uploaded = results["upload_sources"]["sources"][0]
    return {
        "source_id": uploaded["id"],
        "title": uploaded["filename"],
        "excerpt": "Evidence traceability links claims to source material so reviewers can evaluate support quality.",
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run a ready demo against every Ollama Claim Analysis API endpoint.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("API_KEY") or _first_api_key_from_env())
    parser.add_argument("--claim", default=DEFAULT_CLAIM)
    parser.add_argument("--target-style", default="conference")
    parser.add_argument("--review-paper-ai", action="store_true", help="Call Ollama for POST /review/paper.")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    results = run_demo(
        DemoConfig(
            base_url=args.base_url,
            claim=args.claim,
            target_style=args.target_style,
            review_paper_use_ai=args.review_paper_ai,
            work_dir=args.work_dir,
            api_key=args.api_key,
        )
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def _first_api_key_from_env() -> str | None:
    raw_api_keys = os.getenv("API_KEYS", "")
    keys = [key.strip() for key in raw_api_keys.split(",") if key.strip()]
    return keys[0] if keys else None


if __name__ == "__main__":
    raise SystemExit(main())
