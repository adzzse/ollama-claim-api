from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Verdict = Literal["supported", "partially_supported", "unsupported", "unclear"]


class ModelSummary(BaseModel):
    name: str


class ModelsResponse(BaseModel):
    models: list[ModelSummary]


class ClaimAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=4000)
    source_id: str = Field(min_length=1, max_length=120)
    title: str | None = Field(default=None, max_length=300)
    excerpt: str = Field(min_length=1, max_length=4000)

    @field_validator("claim", "source_id", "excerpt")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be empty")
        return stripped

    @field_validator("title")
    @classmethod
    def strip_optional_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ClaimAnalysisResponse(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    matched_source_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1, max_length=2000)

    @field_validator("matched_source_ids", "missing_evidence")
    @classmethod
    def strip_list_items(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @field_validator("explanation")
    @classmethod
    def strip_explanation(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("explanation must not be empty")
        return stripped


class ExtractResponse(BaseModel):
    filename: str
    method: Literal["mineru", "text"]
    markdown: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12000)

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must not be empty")
        return stripped


class GenerateResponse(BaseModel):
    model: str = Field(min_length=1)
    response: str = Field(min_length=1)
    done: bool

    @field_validator("model", "response")
    @classmethod
    def strip_generate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be empty")
        return stripped


class EmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class EmbeddingResponse(BaseModel):
    embedding: list[float]


class ChunkEmbedding(BaseModel):
    chunkIndex: int
    text: str
    embedding: list[float]


class ProcessDocumentResponse(BaseModel):
    status: Literal["SUCCESS"]
    data: list[ChunkEmbedding]


class ProcessDocumentErrorResponse(BaseModel):
    status: Literal["ERROR"]
    detail: str
