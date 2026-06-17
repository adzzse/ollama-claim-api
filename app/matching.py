import logging
import math
import re
from collections import Counter

from app.models import ClaimMatch
from app.store import SourceChunk


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


logger = logging.getLogger(__name__)


def match_claim_to_chunks(claim: str, chunks: list[SourceChunk], top_k: int) -> list[ClaimMatch]:
    claim_terms = _tokens(claim)
    logger.debug(
        "score chunks start claim_terms=%s candidate_chunks=%s top_k=%s",
        len(claim_terms),
        len(chunks),
        top_k,
    )
    scored = []
    for chunk in chunks:
        score = _score(claim_terms, _tokens(chunk.text))
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    matches = [
        ClaimMatch(
            source_id=chunk.source_id,
            filename=chunk.filename,
            chunk_id=chunk.id,
            page=chunk.page,
            excerpt=chunk.text,
            score=round(score, 3),
            suitability=_suitability(score),
            explanation=_explanation(score),
        )
        for score, chunk in scored[:top_k]
    ]
    logger.debug("score chunks complete scored_chunks=%s returned_matches=%s", len(scored), len(matches))
    return matches


def _tokens(text: str) -> Counter[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
    return Counter(word for word in words if len(word) > 2 and word not in STOPWORDS)


def _score(claim_terms: Counter[str], chunk_terms: Counter[str]) -> float:
    if not claim_terms or not chunk_terms:
        return 0.0
    overlap = set(claim_terms) & set(chunk_terms)
    if not overlap:
        return 0.0
    weighted_overlap = sum(min(claim_terms[word], chunk_terms[word]) for word in overlap)
    denominator = math.sqrt(sum(claim_terms.values())) * math.sqrt(sum(chunk_terms.values()))
    return min(weighted_overlap / denominator, 1.0)


def _suitability(score: float) -> str:
    if score >= 0.45:
        return "strong"
    if score >= 0.25:
        return "medium"
    return "weak"


def _explanation(score: float) -> str:
    if score >= 0.45:
        return "This source chunk shares strong terminology with the claim and is a good candidate for review."
    if score >= 0.25:
        return "This source chunk has partial overlap with the claim and may support part of it."
    return "This source chunk has limited overlap with the claim and should be checked manually."
