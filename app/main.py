import logging

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.extraction import ExtractionError, extract_upload_text
from app.logging_config import configure_app_logging
from app.matching import match_claim_to_chunks
from app.models import (
    ClaimAnalysisRequest,
    ClaimAnalysisResponse,
    ClaimMatchRequest,
    ClaimMatchResponse,
    PaperReviewRequest,
    PaperReviewResponse,
    PaperSummary,
    PaperUploadResponse,
    SourceListResponse,
    SourceReferenceSummary,
    SourceReferencesResponse,
    SourceSummary,
    SourceUploadResponse,
)
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    analyze_paper_review,
    analyze_claim,
    check_ollama,
)
from app.paper_review import review_paper
from app.settings import Settings, load_settings
from app.store import demo_store


configure_app_logging()
app = FastAPI(title="Ollama Claim Analysis API", version="0.2.0")
logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return load_settings()


async def check_ollama_health(
    settings: Settings = Depends(get_settings),
) -> dict:
    return await check_ollama(settings)


async def run_claim_analysis(
    payload: ClaimAnalysisRequest,
    settings: Settings = Depends(get_settings),
) -> ClaimAnalysisResponse:
    return await analyze_claim(payload, settings)


@app.get("/health")
async def health(
    settings: Settings = Depends(get_settings),
    ollama_status: dict = Depends(check_ollama_health),
) -> dict:
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "ollama": ollama_status,
    }


@app.post("/process/claim", response_model=ClaimAnalysisResponse)
async def process_claim(
    result: ClaimAnalysisResponse = Depends(run_claim_analysis),
) -> ClaimAnalysisResponse:
    return result


@app.post("/sources", response_model=SourceUploadResponse)
async def upload_sources(
    files: list[UploadFile] = File(...),
) -> SourceUploadResponse:
    summaries: list[SourceSummary] = []
    logger.debug("upload sources start file_count=%s", len(files))
    for file in files:
        filename = file.filename or "uploaded-source"
        logger.debug(
            "upload source start filename=%s content_type=%s",
            filename,
            file.content_type,
        )
        text = await extract_upload_text(file)
        record = demo_store.add_source(filename, text)
        logger.debug(
            "upload source complete source_id=%s filename=%s text_chars=%s chunks=%s references=%s",
            record.id,
            record.filename,
            len(record.text),
            len(record.chunks),
            len(record.references),
        )
        summaries.append(
            SourceSummary(
                id=record.id,
                filename=record.filename,
                chunk_count=len(record.chunks),
                reference_count=len(record.references),
            )
        )
    return SourceUploadResponse(sources=summaries)


@app.get("/sources", response_model=SourceListResponse)
async def list_sources() -> SourceListResponse:
    return SourceListResponse(
        sources=[
            SourceSummary(
                id=source.id,
                filename=source.filename,
                chunk_count=len(source.chunks),
                reference_count=len(source.references),
            )
            for source in demo_store.sources.values()
        ]
    )


@app.get("/sources/references", response_model=SourceReferencesResponse)
async def list_source_references() -> SourceReferencesResponse:
    return SourceReferencesResponse(
        references=[
            SourceReferenceSummary(
                id=reference.id,
                source_id=reference.source_id,
                filename=reference.filename,
                raw_text=reference.raw_text,
                title=reference.title,
                year=reference.year,
            )
            for reference in demo_store.all_references()
        ]
    )


@app.post("/match/claim", response_model=ClaimMatchResponse)
async def match_claim(
    payload: ClaimMatchRequest,
) -> ClaimMatchResponse:
    chunks = demo_store.all_chunks()
    logger.debug(
        "match claim start claim_chars=%s available_chunks=%s top_k=%s",
        len(payload.claim),
        len(chunks),
        payload.top_k,
    )
    if not chunks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No sources have been ingested")

    matches = match_claim_to_chunks(payload.claim, chunks, payload.top_k)
    logger.debug(
        "match claim complete claim_chars=%s available_chunks=%s returned_matches=%s",
        len(payload.claim),
        len(chunks),
        len(matches),
    )
    return ClaimMatchResponse(claim=payload.claim, matches=matches)


@app.post("/papers", response_model=PaperUploadResponse)
async def upload_paper(
    file: UploadFile = File(...),
) -> PaperUploadResponse:
    filename = file.filename or "uploaded-paper"
    logger.debug("upload paper start filename=%s content_type=%s", filename, file.content_type)
    text = await extract_upload_text(file)
    record = demo_store.add_paper(filename, text)
    logger.debug(
        "upload paper complete paper_id=%s filename=%s text_chars=%s sections=%s",
        record.id,
        record.filename,
        len(record.text),
        len(record.sections),
    )
    return PaperUploadResponse(
        paper=PaperSummary(id=record.id, filename=record.filename, section_count=len(record.sections))
    )


@app.post("/review/paper", response_model=PaperReviewResponse)
async def review_uploaded_paper(
    payload: PaperReviewRequest,
    settings: Settings = Depends(get_settings),
) -> PaperReviewResponse:
    logger.debug(
        "review paper start paper_id=%s target_style=%s use_ai=%s",
        payload.paper_id,
        payload.target_style,
        payload.use_ai,
    )
    record = demo_store.papers.get(payload.paper_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")

    review = review_paper(record, payload.target_style)
    if payload.use_ai:
        logger.info("review paper ai enabled paper_id=%s use_ai=true", payload.paper_id)
        review = await analyze_paper_review(record, payload.target_style, review, settings)
    else:
        logger.info("review paper ai skipped paper_id=%s use_ai=false", payload.paper_id)
    logger.debug(
        "review paper complete paper_id=%s detected_style=%s target_style=%s missing=%s weak=%s claim_recommendations=%s",
        payload.paper_id,
        review["detected_style"],
        review["target_style"],
        len(review["missing_sections"]),
        len(review["weak_sections"]),
        len(review["claim_recommendations"]),
    )
    return PaperReviewResponse.model_validate(review)


@app.exception_handler(OllamaUnavailableError)
async def ollama_unavailable_handler(_, exc: OllamaUnavailableError):
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


@app.exception_handler(OllamaInvalidResponseError)
async def ollama_invalid_response_handler(_, exc: OllamaInvalidResponseError):
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=str(exc),
    )


@app.exception_handler(ExtractionError)
async def extraction_error_handler(_, exc: ExtractionError):
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=str(exc),
    )
