from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


DocumentFormat: TypeAlias = Literal[
    "html",
    "pdf",
    "docx",
    "xlsx",
    "pptx",
]
ParsingFailureCode: TypeAlias = Literal[
    "parser-not-registered",
    "format-mismatch",
    "document-read-failed",
    "unsupported-text-encoding",
    "no-extractable-content",
    "invalid-document-package",
    "invalid-pdf",
    "invalid-spreadsheet",
    "invalid-presentation",
    "password-protected-document",
    "pdf-text-layer-missing",
    "invalid-parser-output",
]

_FORMAT_BY_EXTENSION: dict[str, DocumentFormat] = {
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
}
_SUPPORTED_DOCUMENT_FORMATS = frozenset(
    _FORMAT_BY_EXTENSION.values()
)


class UnsupportedDocumentFormatError(ValueError):
    pass


class DocumentParsingError(ValueError):
    failure_code: ParsingFailureCode
    retryable: bool = False
    operator_action: str


class DocumentFormatMismatchError(DocumentParsingError):
    failure_code = "format-mismatch"
    operator_action = "investigate-format-routing"


class DocumentReadError(DocumentParsingError):
    failure_code = "document-read-failed"
    retryable = True
    operator_action = "retry-source-read"


class UnsupportedTextEncodingError(DocumentParsingError):
    failure_code = "unsupported-text-encoding"
    operator_action = "detect-or-transcode-encoding"


class NoExtractableContentError(DocumentParsingError):
    failure_code = "no-extractable-content"
    operator_action = "use-fallback-extractor-or-manual-review"


class InvalidDocumentPackageError(DocumentParsingError):
    failure_code = "invalid-document-package"
    operator_action = "redownload-or-manual-review"


class InvalidPdfError(DocumentParsingError):
    failure_code = "invalid-pdf"
    operator_action = "redownload-or-manual-review"


class InvalidSpreadsheetError(DocumentParsingError):
    failure_code = "invalid-spreadsheet"
    operator_action = "redownload-or-manual-review"


class InvalidPresentationError(DocumentParsingError):
    failure_code = "invalid-presentation"
    operator_action = "redownload-or-manual-review"


class PasswordProtectedDocumentError(DocumentParsingError):
    failure_code = "password-protected-document"
    operator_action = "request-unlocked-document-or-manual-review"


class PdfTextLayerMissingError(DocumentParsingError):
    failure_code = "pdf-text-layer-missing"
    operator_action = "run-ocr-or-manual-review"


@dataclass(frozen=True, slots=True)
class DocumentArtifact:
    path: Path
    source_reference: str
    document_format: DocumentFormat
    content_sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if not self.source_reference.strip():
            raise ValueError("source_reference must not be blank")
        if self.document_format not in _SUPPORTED_DOCUMENT_FORMATS:
            raise ValueError("document_format is not supported")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 hex digest")
        try:
            int(self.content_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "content_sha256 must be a SHA-256 hex digest"
            ) from error
        if self.byte_size < 0:
            raise ValueError("byte_size must not be negative")


@dataclass(frozen=True, slots=True)
class EvidenceLocation:
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()
    sheet_name: str | None = None
    cell_reference: str | None = None
    slide_number: int | None = None
    table_number: int | None = None
    table_row_number: int | None = None

    def __post_init__(self) -> None:
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive")
        if self.slide_number is not None and self.slide_number < 1:
            raise ValueError("slide_number must be positive")
        if self.table_number is not None and self.table_number < 1:
            raise ValueError("table_number must be positive")
        if (
            self.table_row_number is not None
            and self.table_row_number < 1
        ):
            raise ValueError("table_row_number must be positive")
        if self.table_row_number is not None and self.table_number is None:
            raise ValueError(
                "table_number is required when table_row_number is set"
            )
        if self.cell_reference and not self.sheet_name:
            raise ValueError(
                "sheet_name is required when cell_reference is set"
            )


@dataclass(frozen=True, slots=True)
class ParsedFragment:
    ordinal: int
    text: str
    location: EvidenceLocation

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        if not self.text.strip():
            raise ValueError("text must not be blank")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    artifact: DocumentArtifact
    fragments: tuple[ParsedFragment, ...]

    def __post_init__(self) -> None:
        if not self.fragments:
            raise ValueError("parsed document must contain fragments")
        actual_ordinals = tuple(
            fragment.ordinal for fragment in self.fragments
        )
        expected_ordinals = tuple(range(len(self.fragments)))
        if actual_ordinals != expected_ordinals:
            raise ValueError(
                "fragment ordinals must be contiguous and start at zero"
            )


def detect_document_format(path: Path) -> DocumentFormat:
    extension = path.suffix.casefold()
    try:
        return _FORMAT_BY_EXTENSION[extension]
    except KeyError as error:
        raise UnsupportedDocumentFormatError(
            f"unsupported document extension: {extension or '<none>'}"
        ) from error


def inspect_document_artifact(
    path: Path,
    source_reference: str,
) -> DocumentArtifact:
    if not path.is_file():
        raise FileNotFoundError(f"document file does not exist: {path}")

    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as document_file:
        while chunk := document_file.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)

    return DocumentArtifact(
        path=path,
        source_reference=source_reference,
        document_format=detect_document_format(path),
        content_sha256=digest.hexdigest(),
        byte_size=byte_size,
    )
