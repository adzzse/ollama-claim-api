from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from fastapi import UploadFile

from app.env import load_runtime_env


logger = logging.getLogger(__name__)


class ExtractionError(ValueError):
    pass


class MinerUUnavailableError(RuntimeError):
    pass


class MinerUExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedMarkdown:
    filename: str
    method: str
    markdown: str


def check_mineru() -> dict:
    load_runtime_env()
    command = os.getenv("MINERU_COMMAND", "mineru")
    resolved = shutil.which(command)
    return {
        "ok": resolved is not None,
        "command": command,
        "resolved": resolved,
        "method": os.getenv("MINERU_METHOD", "auto"),
        "backend": os.getenv("MINERU_BACKEND"),
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
            markdown = extract_with_mineru(filename, raw)
        except (MinerUUnavailableError, MinerUExtractionError) as exc:
            raise ExtractionError(str(exc)) from exc
        logger.debug("extract markdown complete filename=%s method=mineru chars=%s", filename, len(markdown))
        return ExtractedMarkdown(filename=filename, method="mineru", markdown=markdown)

    raise ExtractionError(f"Unsupported file type for {filename}")


def extract_with_mineru(filename: str, raw: bytes) -> str:
    load_runtime_env()
    command = os.getenv("MINERU_COMMAND", "mineru")
    if shutil.which(command) is None:
        raise MinerUUnavailableError(f"MinerU command is not available: {command}")

    timeout_seconds = int(os.getenv("MINERU_TIMEOUT_SECONDS", "600"))
    suffix = Path(filename).suffix.lower() or ".pdf"

    with tempfile.TemporaryDirectory(prefix="evidencepilot-mineru-") as temp_dir:
        work_dir = Path(temp_dir)
        input_path = work_dir / f"input{suffix}"
        output_dir = work_dir / "output"
        input_path.write_bytes(raw)
        output_dir.mkdir()

        args = [
            command,
            "--path",
            str(input_path),
            "--output",
            str(output_dir),
            "--method",
            os.getenv("MINERU_METHOD", "auto"),
        ]
        if api_url := os.getenv("MINERU_API_URL"):
            args.extend(["--api-url", api_url])
        if backend := os.getenv("MINERU_BACKEND"):
            args.extend(["--backend", backend])

        logger.debug(
            "mineru run start filename=%s command=%s method=%s timeout_seconds=%s",
            filename,
            command,
            os.getenv("MINERU_METHOD", "auto"),
            timeout_seconds,
        )
        returncode, stdout, stderr = _run_mineru_process(args, timeout_seconds)

        if returncode != 0:
            detail = (stderr or stdout).strip()
            message = f"MinerU extraction failed: {detail}" if detail else "MinerU extraction failed"
            raise MinerUExtractionError(message)

        logger.debug("mineru run complete filename=%s output_dir=%s", filename, output_dir)
        return _read_mineru_markdown(output_dir)


def _run_mineru_process(args: list[str], timeout_seconds: int) -> tuple[int, str, str]:
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise MinerUExtractionError("MinerU extraction timed out") from exc

    returncode = process.returncode
    return returncode, stdout or "", stderr or ""


def _read_mineru_markdown(output_dir: Path) -> str:
    markdown_files = sorted(
        (path for path in output_dir.rglob("*.md") if path.is_file()),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            logger.debug(
                "mineru markdown selected path=%s bytes=%s",
                markdown_file,
                markdown_file.stat().st_size,
            )
            return _clean_text(text)
    raise MinerUExtractionError("MinerU did not produce Markdown text")


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
