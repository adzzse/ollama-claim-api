import re

from app.store import SourceReference


REFERENCE_HEADINGS = {"references", "bibliography", "works cited"}
REFERENCE_PREFIX = re.compile(r"^\s*(?:\[\d+\]|\d+[\).]|[-*])\s*")
YEAR_PATTERN = re.compile(r"\((?P<paren>(?:18|19|20)\d{2})\)|\b(?P<plain>(?:18|19|20)\d{2})\b")


def extract_references_for_demo(text: str, source_id: str, filename: str) -> list[SourceReference]:
    entries = _reference_entries(text)
    return [
        SourceReference(
            id=f"{source_id}-ref-{index + 1}",
            source_id=source_id,
            filename=filename,
            raw_text=entry,
            title=_extract_title(entry),
            year=_extract_year(entry),
        )
        for index, entry in enumerate(entries)
    ]


def _reference_entries(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    start_index = _references_start_index(lines)
    if start_index is None:
        return []

    entries: list[str] = []
    current = ""
    for line in lines[start_index:]:
        if not line:
            if current:
                entries.append(current)
                current = ""
            continue
        if _is_markdown_heading(line):
            break
        if _starts_reference(line) or (current and _looks_like_unnumbered_reference_start(line)):
            if current:
                entries.append(current)
            current = _strip_reference_prefix(line)
        elif current:
            current = f"{current} {line}".strip()
        else:
            current = _strip_reference_prefix(line)

    if current:
        entries.append(current)
    return entries


def _references_start_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        normalized = _normalize_heading(line)
        if normalized in REFERENCE_HEADINGS:
            return index + 1
    return None


def _normalize_heading(line: str) -> str:
    return line.lstrip("#").strip().lower().rstrip(":")


def _is_markdown_heading(line: str) -> bool:
    return line.startswith("#") and bool(line.lstrip("#").strip())


def _starts_reference(line: str) -> bool:
    return bool(REFERENCE_PREFIX.match(line))


def _looks_like_unnumbered_reference_start(line: str) -> bool:
    year_match = YEAR_PATTERN.search(line)
    if not year_match:
        return False
    return year_match.start() <= 600 and "," in line[: year_match.start()]


def _strip_reference_prefix(line: str) -> str:
    return REFERENCE_PREFIX.sub("", line).strip()


def _extract_year(reference: str) -> int | None:
    match = YEAR_PATTERN.search(reference)
    if not match:
        return None
    return int(match.group("paren") or match.group("plain"))


def _extract_title(reference: str) -> str | None:
    quoted = re.search(r'"([^"]+)"', reference)
    if quoted:
        return quoted.group(1).strip() or None

    year_match = YEAR_PATTERN.search(reference)
    if year_match:
        rest = reference[year_match.end() :].lstrip(". ")
        title = rest.split(".", 1)[0].strip()
        return title or None

    parts = [part.strip() for part in reference.split(".") if part.strip()]
    if len(parts) >= 2:
        return parts[1]
    return None
