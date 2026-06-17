from app.models import SectionIssue
from app.store import PaperRecord, PaperSection


STYLE_RULES = {
    "conference": [
        "Abstract",
        "Introduction",
        "Related Work",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
        "References",
    ],
    "article": ["Title", "Introduction", "Main Argument", "Evidence", "Conclusion"],
    "magazine": ["Lead", "Main Story", "Examples", "Takeaway"],
    "report": ["Executive Summary", "Background", "Findings", "Recommendations", "Conclusion"],
    "thesis": ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion", "References"],
}


def review_paper(record: PaperRecord, target_style: str | None) -> dict:
    detected_style = detect_style(record.sections)
    chosen_style = target_style or detected_style
    if chosen_style == "unknown":
        chosen_style = "article"

    expected_sections = STYLE_RULES.get(chosen_style, STYLE_RULES["article"])
    present_names = {_normalize(section.name) for section in record.sections}

    missing = [
        SectionIssue(
            section=name,
            issue=f"{name} is expected for a {chosen_style} paper but was not found.",
            recommendation=f"Add a {name} section or rename the matching content clearly.",
        )
        for name in expected_sections
        if _normalize(name) not in present_names and not _has_alias(name, present_names)
    ]

    weak = [
        SectionIssue(
            section=section.name,
            issue="This section is short for its role in the paper.",
            recommendation="Add clearer claims, supporting explanation, or evidence references.",
        )
        for section in record.sections
        if len(section.text.split()) < 18
    ]

    claim_recommendations = _claim_recommendations(record.sections, chosen_style)

    return {
        "paper_id": record.id,
        "detected_style": detected_style,
        "target_style": chosen_style,
        "missing_sections": missing,
        "weak_sections": weak,
        "claim_recommendations": claim_recommendations,
    }


def detect_style(sections: list[PaperSection]) -> str:
    present = {_normalize(section.name) for section in sections}
    best_style = "unknown"
    best_score = 0
    for style, expected in STYLE_RULES.items():
        expected_normalized = {_normalize(name) for name in expected}
        score = len(present & expected_normalized)
        if score > best_score:
            best_style = style
            best_score = score
    return best_style if best_score else "unknown"


def _claim_recommendations(sections: list[PaperSection], style: str) -> list[SectionIssue]:
    recommendations: list[SectionIssue] = []
    claim_words = {"claim", "argue", "shows", "demonstrates", "evidence", "support", "because"}
    for section in sections:
        section_text = section.text.lower()
        if _normalize(section.name) in {"introduction", "related work", "discussion", "main argument", "findings"}:
            if not any(word in section_text for word in claim_words):
                recommendations.append(
                    SectionIssue(
                        section=section.name,
                        issue="This section does not state enough explicit claims or evidence signals.",
                        recommendation=f"Add 1-2 clear claims that fit the {style} style and connect them to source evidence.",
                    )
                )

    if not recommendations:
        recommendations.append(
            SectionIssue(
                section="Overall",
                issue="Claim coverage should be checked against the paper's main argument.",
                recommendation="Review each major section and ensure important assertions are tied to evidence.",
            )
        )
    return recommendations


def _normalize(value: str) -> str:
    return value.lower().strip().replace("_", " ")


def _has_alias(expected: str, present_names: set[str]) -> bool:
    aliases = {
        "methodology": {"methods"},
        "main story": {"body"},
        "takeaway": {"conclusion"},
    }
    return bool(aliases.get(_normalize(expected), set()) & present_names)
