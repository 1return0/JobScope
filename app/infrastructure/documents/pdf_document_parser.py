from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    DocumentFormat,
    DocumentFormatMismatchError,
    DocumentReadError,
    EvidenceLocation,
    InvalidPdfError,
    ParsedDocument,
    ParsedFragment,
    PasswordProtectedDocumentError,
    PdfTextLayerMissingError,
)


class PdfDocumentParser:
    document_format: DocumentFormat = "pdf"

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        if artifact.document_format != self.document_format:
            raise DocumentFormatMismatchError(
                "PdfDocumentParser only accepts pdf artifacts"
            )

        reader = self._open_reader(artifact.path)
        fragments: list[ParsedFragment] = []

        try:
            for page_number, page in enumerate(reader.pages, start=1):
                text = self._normalize_text(page.extract_text() or "")
                if not text:
                    continue
                fragments.append(
                    ParsedFragment(
                        ordinal=len(fragments),
                        text=text,
                        location=EvidenceLocation(
                            page_number=page_number,
                        ),
                    )
                )
        except FileNotDecryptedError as error:
            raise PasswordProtectedDocumentError(
                "PDF requires a password before text can be extracted"
            ) from error
        except (PdfReadError, ValueError, KeyError) as error:
            raise InvalidPdfError(
                "PDF content could not be parsed"
            ) from error

        if not fragments:
            raise PdfTextLayerMissingError(
                "PDF contains no extractable text layer"
            )

        return ParsedDocument(
            artifact=artifact,
            fragments=tuple(fragments),
        )

    @staticmethod
    def _open_reader(path: Path) -> PdfReader:
        if not path.is_file():
            raise DocumentReadError(
                f"PDF document could not be read: {path}"
            )
        try:
            reader = PdfReader(path)
        except OSError as error:
            raise DocumentReadError(
                f"PDF document could not be read: {path}"
            ) from error
        except (PdfReadError, ValueError) as error:
            raise InvalidPdfError(
                "PDF file is not a valid readable document"
            ) from error

        if reader.is_encrypted:
            try:
                decryption_result = reader.decrypt("")
            except (FileNotDecryptedError, PdfReadError) as error:
                raise PasswordProtectedDocumentError(
                    "PDF requires a password before it can be parsed"
                ) from error
            if not decryption_result:
                raise PasswordProtectedDocumentError(
                    "PDF requires a password before it can be parsed"
                )

        return reader

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())
