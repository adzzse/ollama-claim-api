import asyncio
import logging
import time

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.extraction import ExtractionError, check_liteparse, extract_upload_markdown
from app.logging_config import configure_app_logging
from app.models import (
    ClaimAnalysisRequest,
    ClaimAnalysisResponse,
    ChunkEmbedding,
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractResponse,
    GenerateRequest,
    GenerateResponse,
    ModelsResponse,
    ProcessDocumentResponse,
)
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    analyze_claim,
    check_ollama,
    generate_embeddings,
    generate_response,
    list_models,
)
from app.settings import Settings, load_settings

EMBED_BATCH_SIZE = 5
EMBED_MAX_RETRIES = 3
EMBED_RETRY_DELAY_SEC = 1.0


configure_app_logging()
app = FastAPI(title="Ollama Claim Analysis API", version="0.2.0")
logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return load_settings()


async def check_ollama_health(
    settings: Settings = Depends(get_settings),
) -> dict:
    return await check_ollama(settings)


def check_liteparse_health() -> dict:
    return check_liteparse()


async def run_claim_analysis(
    payload: ClaimAnalysisRequest,
    settings: Settings = Depends(get_settings),
) -> ClaimAnalysisResponse:
    return await analyze_claim(payload, settings)


@app.get("/health")
async def health(
    settings: Settings = Depends(get_settings),
    ollama_status: dict = Depends(check_ollama_health),
    liteparse_status: dict = Depends(check_liteparse_health),
) -> dict:
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "ollama": ollama_status,
        "liteparse": liteparse_status,
    }


@app.get("/ai/models", response_model=ModelsResponse)
async def ai_models(
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    return await list_models(settings)


@app.post("/process/claim", response_model=ClaimAnalysisResponse)
async def process_claim(
    result: ClaimAnalysisResponse = Depends(run_claim_analysis),
) -> ClaimAnalysisResponse:
    return result


@app.post("/extract", response_model=ExtractResponse)
async def extract_document(
    file: UploadFile = File(...),
) -> ExtractResponse:
    result = await extract_upload_markdown(file)
    return ExtractResponse(
        filename=result.filename,
        method=result.method,
        markdown=result.markdown,
    )


async def run_generate_embeddings(
    payload: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> EmbeddingResponse:
    logger.debug(
        "embeddings start text_chars=%s model=%s",
        len(payload.text),
        settings.ollama_embedding_model,
    )
    vector = await generate_embeddings(payload.text, settings)
    logger.debug(
        "embeddings complete text_chars=%s vector_dim=%s",
        len(payload.text),
        len(vector),
    )
    return EmbeddingResponse(embedding=vector)


async def run_generate_text(
    payload: GenerateRequest,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    logger.debug(
        "generate start prompt_chars=%s model=%s",
        len(payload.prompt),
        settings.ollama_model,
    )
    result = GenerateResponse.model_validate(await generate_response(payload, settings))
    logger.debug(
        "generate complete prompt_chars=%s response_chars=%s",
        len(payload.prompt),
        len(result.response),
    )
    return result


@app.post("/ai/generate", response_model=GenerateResponse)
async def generate(
    result: GenerateResponse = Depends(run_generate_text),
) -> GenerateResponse:
    return result


@app.post("/ai/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    result: EmbeddingResponse = Depends(run_generate_embeddings),
) -> EmbeddingResponse:
    return result


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    fences = _code_fence_ranges(text)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            fence = _fence_containing(fences, start, end)
            if fence is not None:
                end = min(fence[1], len(text))
            else:
                para = text.rfind("\n\n", start, end)
                if para > start + chunk_size // 2:
                    end = para + 2
                else:
                    nl = text.rfind("\n", start, end)
                    if nl > start + chunk_size // 2:
                        end = nl + 1
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def _code_fence_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    search_start = 0
    while True:
        open_idx = text.find("```", search_start)
        if open_idx == -1:
            break
        close_idx = text.find("```", open_idx + 3)
        if close_idx == -1:
            break
        ranges.append((open_idx, close_idx + 3))
        search_start = close_idx + 3
    return ranges


def _fence_containing(fences: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    for f_start, f_end in fences:
        if start >= f_start and start < f_end:
            return (f_start, f_end)
        if start < f_start and end > f_start:
            return (f_start, f_end)
    return None


async def _embed_with_retry(index: int, text: str, settings: Settings) -> ChunkEmbedding:
    last_exc: Exception | None = None
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            vector = await generate_embeddings(text, settings)
            return ChunkEmbedding(chunkIndex=index, text=text, embedding=vector)
        except (OllamaUnavailableError, OllamaInvalidResponseError) as exc:
            last_exc = exc
            logger.warning(
                "embedding retry %d/%d for chunk %d: %s",
                attempt,
                EMBED_MAX_RETRIES,
                index,
                exc,
            )
            if attempt < EMBED_MAX_RETRIES:
                await asyncio.sleep(EMBED_RETRY_DELAY_SEC * attempt)
    raise last_exc  # type: ignore[misc]


@app.post("/ai/process-document")
async def process_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    try:
        extracted = await extract_upload_markdown(file)
    except ExtractionError as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "ERROR", "detail": str(exc)},
        )

    raw_chunks = _chunk_text(extracted.markdown)
    all_results: list[ChunkEmbedding] = []

    for batch_start in range(0, len(raw_chunks), EMBED_BATCH_SIZE):
        batch = raw_chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        batch_results = await asyncio.gather(
            *(_embed_with_retry(batch_start + i, t, settings) for i, t in enumerate(batch)),
            return_exceptions=True,
        )
        for result in batch_results:
            if isinstance(result, Exception):
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"status": "ERROR", "detail": str(result)},
                )
            all_results.append(result)

    return ProcessDocumentResponse(status="SUCCESS", data=all_results)


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
