import json
import logging
from typing import Any

import httpx
from pydantic import ValidationError

from app.models import ClaimAnalysisRequest, ClaimAnalysisResponse, PaperReviewResponse
from app.settings import Settings
from app.store import PaperRecord


logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaInvalidResponseError(RuntimeError):
    pass


def build_claim_prompt(payload: ClaimAnalysisRequest) -> str:
    source = (
        f"- id: {payload.source_id}\n"
        f"  title: {payload.title or 'Untitled source'}\n"
        f"  excerpt: {payload.excerpt}"
    )
    return f"""You are an evidence reviewer for academic writing.
Judge whether the claim is supported only by the provided source excerpts.
Do not use outside knowledge. Do not invent sources.

Return strict JSON only, with exactly these keys:
- verdict: one of "supported", "partially_supported", "unsupported", "unclear"
- confidence: number from 0.0 to 1.0
- matched_source_ids: array of source ids that support the verdict
- missing_evidence: array of concise notes about gaps or needed evidence
- reasoning_summary: 2-4 concise sentences explaining the evidence comparison process, without hidden chain-of-thought
- explanation: concise explanation under 120 words

Claim:
{payload.claim}

Sources:
{source}
"""


async def check_ollama(settings: Settings) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        raise OllamaUnavailableError("Ollama is not reachable") from exc

    data = response.json()
    model_names = {model.get("name") for model in data.get("models", [])}
    return {
        "ok": True,
        "model_available": settings.ollama_model in model_names,
    }


async def analyze_claim(payload: ClaimAnalysisRequest, settings: Settings) -> ClaimAnalysisResponse:
    prompt = build_claim_prompt(payload)
    request_body = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0},
    }
    logger.info(
        "ai request start model=%s claim_chars=%s source_count=%s",
        settings.ollama_model,
        len(payload.claim),
        1,
    )
    logger.debug("ai prompt %s", prompt)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=request_body,
            )
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.info("ai request failed model=%s reason=%s", settings.ollama_model, exc)
        raise OllamaUnavailableError("Ollama generation failed") from exc

    try:
        response_data = response.json()
        raw_model_text = response_data["response"]
        logger.debug("ai raw response %s", raw_model_text)
        if not raw_model_text.strip():
            logger.info(
                "ai response empty done_reason=%s thinking_chars=%s",
                response_data.get("done_reason"),
                len(response_data.get("thinking") or ""),
            )
        model_json = json.loads(raw_model_text)
        reasoning_summary = model_json.get("reasoning_summary")
        if reasoning_summary:
            logger.debug("ai reasoning summary %s", reasoning_summary)
        result = ClaimAnalysisResponse.model_validate(model_json)
        logger.info(
            "ai verdict verdict=%s confidence=%s matched_source_ids=%s missing_evidence=%s",
            result.verdict,
            result.confidence,
            result.matched_source_ids,
            result.missing_evidence,
        )
        logger.info("ai explanation %s", result.explanation)
        return result
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        logger.info("ai response invalid reason=%s", exc)
        raise OllamaInvalidResponseError("Ollama returned invalid claim-analysis JSON") from exc


def build_paper_review_prompt(record: PaperRecord, target_style: str | None, baseline_review: dict) -> str:
    sections = "\n\n".join(
        f"## {section.name}\n{section.text[:1800]}"
        for section in record.sections
    )
    return f"""You are an academic paper reviewer for EvidencePilot.
Review the uploaded paper using only the paper sections and baseline rule findings below.
Do not invent sections that are not implied by the paper. Keep recommendations concrete and student-facing.

Return strict JSON only, with exactly these keys:
- paper_id: "{record.id}"
- detected_style: one of "conference", "article", "magazine", "report", "thesis", "unknown"
- target_style: one of "conference", "article", "magazine", "report", "thesis", "unknown"
- missing_sections: array of objects with section, issue, recommendation
- weak_sections: array of objects with section, issue, recommendation
- claim_recommendations: array of objects with section, issue, recommendation
- reasoning_summary: 2-4 concise sentences explaining how you compared the paper against the target style, without hidden chain-of-thought

Requested target style:
{target_style or "auto-detect"}

Baseline rule review:
{json.dumps(_review_for_prompt(baseline_review), ensure_ascii=False)}

Paper sections:
{sections}
"""


async def analyze_paper_review(
    record: PaperRecord,
    target_style: str | None,
    baseline_review: dict,
    settings: Settings,
) -> dict:
    prompt = build_paper_review_prompt(record, target_style, baseline_review)
    request_body = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0},
    }
    logger.info(
        "paper ai request start model=%s paper_id=%s sections=%s target_style=%s",
        settings.ollama_model,
        record.id,
        len(record.sections),
        target_style,
    )
    logger.debug("paper ai prompt %s", prompt)

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_paper_timeout_seconds) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=request_body,
            )
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.info(
            "paper ai request failed model=%s exception_type=%s reason=%r timeout_seconds=%s",
            settings.ollama_model,
            type(exc).__name__,
            str(exc),
            settings.ollama_paper_timeout_seconds,
        )
        raise OllamaUnavailableError("Ollama paper review failed") from exc

    try:
        response_data = response.json()
        raw_model_text = response_data["response"]
        logger.debug("paper ai raw response %s", raw_model_text)
        if not raw_model_text.strip():
            logger.info(
                "paper ai response empty done_reason=%s thinking_chars=%s",
                response_data.get("done_reason"),
                len(response_data.get("thinking") or ""),
            )
        model_json = json.loads(raw_model_text)
        reasoning_summary = model_json.get("reasoning_summary")
        if reasoning_summary:
            logger.debug("paper ai reasoning summary %s", reasoning_summary)
        model_json["paper_id"] = record.id
        result = PaperReviewResponse.model_validate(model_json)
        logger.info(
            "paper ai review complete paper_id=%s detected_style=%s target_style=%s missing=%s weak=%s claim_recommendations=%s",
            result.paper_id,
            result.detected_style,
            result.target_style,
            len(result.missing_sections),
            len(result.weak_sections),
            len(result.claim_recommendations),
        )
        return result.model_dump()
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        logger.info("paper ai response invalid reason=%s", exc)
        raise OllamaInvalidResponseError("Ollama returned invalid paper-review JSON") from exc


def _review_for_prompt(review: dict) -> dict:
    return {
        key: [_issue_for_prompt(issue) for issue in value]
        if isinstance(value, list)
        else value
        for key, value in review.items()
    }


def _issue_for_prompt(issue) -> dict:
    if hasattr(issue, "model_dump"):
        return issue.model_dump()
    return dict(issue)
