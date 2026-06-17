from dataclasses import dataclass, field
import logging


logger = logging.getLogger(__name__)


@dataclass
class SourceChunk:
    id: str
    source_id: str
    filename: str
    text: str
    page: int | None = None


@dataclass
class SourceReference:
    id: str
    source_id: str
    filename: str
    raw_text: str
    title: str | None = None
    year: int | None = None


@dataclass
class SourceRecord:
    id: str
    filename: str
    text: str
    chunks: list[SourceChunk] = field(default_factory=list)
    references: list[SourceReference] = field(default_factory=list)


@dataclass
class PaperSection:
    name: str
    text: str


@dataclass
class PaperRecord:
    id: str
    filename: str
    text: str
    sections: list[PaperSection]


class DemoStore:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.sources: dict[str, SourceRecord] = {}
        self.papers: dict[str, PaperRecord] = {}
        self._source_counter = 0
        self._paper_counter = 0

    def next_source_id(self) -> str:
        self._source_counter += 1
        return f"source-{self._source_counter}"

    def next_paper_id(self) -> str:
        self._paper_counter += 1
        return f"paper-{self._paper_counter}"

    def add_source(self, filename: str, text: str) -> SourceRecord:
        from app.references import extract_references_for_demo

        source_id = self.next_source_id()
        logger.debug("store source start source_id=%s filename=%s text_chars=%s", source_id, filename, len(text))
        chunks = [
            SourceChunk(
                id=f"{source_id}-chunk-{index + 1}",
                source_id=source_id,
                filename=filename,
                text=chunk_text,
            )
            for index, chunk_text in enumerate(chunk_text_for_demo(text))
        ]
        references = extract_references_for_demo(text, source_id, filename)
        record = SourceRecord(id=source_id, filename=filename, text=text, chunks=chunks, references=references)
        self.sources[source_id] = record
        logger.debug(
            "stored source source_id=%s filename=%s chunks=%s references=%s",
            source_id,
            filename,
            len(chunks),
            len(references),
        )
        return record

    def add_paper(self, filename: str, text: str) -> PaperRecord:
        paper_id = self.next_paper_id()
        logger.debug("store paper start paper_id=%s filename=%s text_chars=%s", paper_id, filename, len(text))
        record = PaperRecord(
            id=paper_id,
            filename=filename,
            text=text,
            sections=detect_sections_for_demo(text),
        )
        self.papers[paper_id] = record
        logger.debug("stored paper paper_id=%s filename=%s sections=%s", paper_id, filename, len(record.sections))
        return record

    def all_chunks(self) -> list[SourceChunk]:
        chunks: list[SourceChunk] = []
        for source in self.sources.values():
            chunks.extend(source.chunks)
        return chunks

    def all_references(self) -> list[SourceReference]:
        references: list[SourceReference] = []
        for source in self.sources.values():
            references.extend(source.references)
        return references


def chunk_text_for_demo(text: str, max_chars: int = 900) -> list[str]:
    logger.debug("chunk text start text_chars=%s max_chars=%s", len(text), max_chars)
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph[:max_chars]
    if current:
        chunks.append(current)
    result = chunks or [text[:max_chars]]
    logger.debug("chunk text complete paragraphs=%s chunks=%s", len(paragraphs), len(result))
    return result


def detect_sections_for_demo(text: str) -> list[PaperSection]:
    logger.debug("detect sections start text_chars=%s", len(text))
    lines = [line.strip() for line in text.splitlines()]
    sections: list[PaperSection] = []
    current_name = "Body"
    current_lines: list[str] = []

    for line in lines:
        if not line:
            continue
        if _looks_like_heading(line):
            if current_lines:
                sections.append(PaperSection(name=current_name, text="\n".join(current_lines).strip()))
            current_name = line.rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(PaperSection(name=current_name, text="\n".join(current_lines).strip()))
    result = sections or [PaperSection(name="Body", text=text.strip())]
    logger.debug("detect sections complete sections=%s", len(result))
    return result


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    if len(words) > 8 or len(line) > 80:
        return False
    known = {
        "abstract",
        "introduction",
        "related work",
        "methodology",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "lead",
        "main story",
        "takeaway",
    }
    return line.lower().rstrip(":") in known or line.istitle() or line.isupper()


demo_store = DemoStore()
