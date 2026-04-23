"""Tests for file_reader utility — read_text_from_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiqyn.utils.file_reader import read_text_from_file, supported_extensions


class TestSupportedExtensions:
    def test_returns_list(self) -> None:
        exts = supported_extensions()
        assert isinstance(exts, list)
        assert len(exts) > 0

    def test_contains_txt(self) -> None:
        assert ".txt" in supported_extensions()


class TestReadTxt:
    def test_reads_txt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Тестовый текст на русском.", encoding="utf-8")
        result = read_text_from_file(f)
        assert result == "Тестовый текст на русском."

    def test_reads_multiline_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "multi.txt"
        content = "Строка 1\nСтрока 2\nСтрока 3"
        f.write_text(content, encoding="utf-8")
        result = read_text_from_file(f)
        assert "Строка 1" in result
        assert "Строка 2" in result

    def test_reads_empty_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert read_text_from_file(f) == ""

    def test_uppercase_extension_handled(self, tmp_path: Path) -> None:
        # .TXT (uppercase) — code does path.suffix.lower() so it reads fine
        f = tmp_path / "TEST.TXT"
        f.write_text("текст", encoding="utf-8")
        result = read_text_from_file(f)
        assert result == "текст"


class TestUnsupportedFormat:
    def test_raises_value_error_for_unknown_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported format"):
            read_text_from_file(f)

    def test_error_message_contains_supported_formats(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.odt"
        f.write_text("data")
        with pytest.raises(ValueError) as exc_info:
            read_text_from_file(f)
        assert ".txt" in str(exc_info.value)

    def test_raises_for_no_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "noext"
        f.write_text("data")
        with pytest.raises(ValueError):
            read_text_from_file(f)


class TestMissingFile:
    def test_raises_file_not_found_for_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            read_text_from_file(f)

    def test_raises_for_missing_docx(self, tmp_path: Path) -> None:
        f = tmp_path / "missing.docx"
        # docx path: ImportError if python-docx missing, otherwise FileNotFoundError
        with pytest.raises((FileNotFoundError, ImportError, Exception)):
            read_text_from_file(f)

    def test_raises_for_missing_pdf(self, tmp_path: Path) -> None:
        f = tmp_path / "missing.pdf"
        with pytest.raises((FileNotFoundError, ImportError, Exception)):
            read_text_from_file(f)
