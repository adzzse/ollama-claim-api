from dataclasses import dataclass
from importlib import util as importlib_util
import logging
from pathlib import Path
import tempfile

from fastapi import UploadFile


logger = logging.getLogger(__name__)


class ExtractionError(ValueError):
    pass


class LiteParseUnavailableError(RuntimeError):
    pass


class LiteParseExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedMarkdown:
    filename: str
    method: str
    markdown: str


def check_liteparse() -> dict:
    available = importlib_util.find_spec("liteparse") is not None
    return {
        "ok": available,
        "package": "liteparse",
    }


async def extract_upload_markdown(file: UploadFile) -> ExtractedMarkdown:
    raw = await file.read()
    filename = file.filename or "uploaded-file"
    suffix = Path(filename).suffix.lower()
    logger.debug(
        "extract markdown start filename=%s suffix=%s content_type=%s bytes=%s",
        filename,
        suffix or "<none>",
        file.content_type,
        len(raw),
    )

    if suffix in {"", ".txt", ".md"} or (file.content_type or "").startswith("text/"):
        markdown = _decode_text(raw)
        logger.debug("extract markdown complete filename=%s method=text chars=%s", filename, len(markdown))
        return ExtractedMarkdown(filename=filename, method="text", markdown=markdown)

    if suffix in {".pdf", ".docx", ".pptx", ".xlsx"}:
        try:
            markdown = extract_with_liteparse(filename, raw)
        except (LiteParseUnavailableError, LiteParseExtractionError) as exc:
            raise ExtractionError(str(exc)) from exc
        logger.debug("extract markdown complete filename=%s method=liteparse chars=%s", filename, len(markdown))
        return ExtractedMarkdown(filename=filename, method="liteparse", markdown=markdown)

    raise ExtractionError(f"Unsupported file type for {filename}")


def extract_with_liteparse(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower() or ".pdf"

    try:
        from liteparse import LiteParse
    except ImportError as exc:
        raise LiteParseUnavailableError("LiteParse package is not available") from exc

    with tempfile.TemporaryDirectory(prefix="evidencepilot-liteparse-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        input_path.write_bytes(raw)

        try:
            logger.debug("liteparse run start filename=%s", filename)
            result = LiteParse(output_format="markdown").parse(str(input_path))
            markdown = getattr(result, "text", None)
        except Exception as exc:
            raise LiteParseExtractionError(f"LiteParse extraction failed: {exc}") from exc

    if not isinstance(markdown, str):
        raise LiteParseExtractionError("LiteParse did not produce Markdown text")
    logger.debug("liteparse run complete filename=%s chars=%s", filename, len(markdown))
    return _clean_text(markdown)


def _decode_text(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    return _clean_text(text)


def _clean_text(text: str) -> str:
    cleaned = "\n".join(line.strip() for line in text.replace("\r\n", "\n").split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line)
    if not cleaned.strip():
        raise ExtractionError("No text could be extracted from uploaded file")
    return cleaned.strip()
