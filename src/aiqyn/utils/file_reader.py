"""Utility for reading text from various file formats."""
from __future__ import annotations

from pathlib import Path


def supported_extensions() -> list[str]:
    """Return list of file extensions supported by read_text_from_file."""
    return [".txt", ".docx", ".pdf"]


def read_text_from_file(path: Path) -> str:
    """Extract plain text from a file.

    Supports .txt, .docx, and .pdf formats.

    Args:
        path: Path to the file to read.

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If the file extension is not supported.
        ImportError: If the required library for the format is not installed.
        FileNotFoundError: If the file does not exist.
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".docx":
        return _read_docx(path)

    if suffix == ".pdf":
        return _read_pdf(path)

    raise ValueError(
        f"Unsupported format: {path.suffix!r}. "
        f"Supported: {', '.join(supported_extensions())}"
    )


def _read_docx(path: Path) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "python-docx is required to read .docx files. "
            "Install it with: uv add python-docx"
        ) from exc

    doc = Document(str(path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def _read_pdf(path: Path) -> str:
    """Extract text from a .pdf file using pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to read .pdf files. "
            "Install it with: uv add pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages.append(page_text)
    return "\n\n".join(pages)
