from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Verdict = Literal["supported", "partially_supported", "unsupported", "unclear"]
PaperStyle = Literal["conference", "article", "magazine", "report", "thesis", "unknown"]


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


class SourceSummary(BaseModel):
    id: str
    filename: str
    chunk_count: int
    reference_count: int


class SourceUploadResponse(BaseModel):
    sources: list[SourceSummary]


class SourceListResponse(BaseModel):
    sources: list[SourceSummary]


class SourceReferenceSummary(BaseModel):
    id: str
    source_id: str
    filename: str
    raw_text: str
    title: str | None = None
    year: int | None = None


class SourceReferencesResponse(BaseModel):
    references: list[SourceReferenceSummary]


class ClaimMatchRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("claim")
    @classmethod
    def strip_match_claim(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("claim must not be empty")
        return stripped


class ClaimMatch(BaseModel):
    source_id: str
    filename: str
    chunk_id: str
    page: int | None = None
    excerpt: str
    score: float = Field(ge=0.0, le=1.0)
    suitability: Literal["strong", "medium", "weak"]
    explanation: str


class ClaimMatchResponse(BaseModel):
    claim: str
    matches: list[ClaimMatch]


class PaperSummary(BaseModel):
    id: str
    filename: str
    section_count: int


class PaperUploadResponse(BaseModel):
    paper: PaperSummary


class PaperReviewRequest(BaseModel):
    paper_id: str = Field(min_length=1)
    target_style: PaperStyle | None = None
    use_ai: bool = False

    @field_validator("paper_id")
    @classmethod
    def strip_paper_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("paper_id must not be empty")
        return stripped


class SectionIssue(BaseModel):
    section: str
    issue: str
    recommendation: str


class PaperReviewResponse(BaseModel):
    paper_id: str
    detected_style: PaperStyle
    target_style: PaperStyle
    missing_sections: list[SectionIssue]
    weak_sections: list[SectionIssue]
    claim_recommendations: list[SectionIssue]


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
