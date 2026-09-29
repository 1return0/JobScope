from __future__ import annotations

import re
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    DocumentFormat,
    DocumentFormatMismatchError,
    DocumentReadError,
    EvidenceLocation,
    InvalidDocumentPackageError,
    NoExtractableContentError,
    ParsedDocument,
    ParsedFragment,
)


_HEADING_STYLE_PATTERN = re.compile(
    r"^(?:heading|标题)\s*([1-6])$",
    re.IGNORECASE,
)


class DocxDocumentParser:
    document_format: DocumentFormat = "docx"

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        if artifact.document_format != self.document_format:
            raise DocumentFormatMismatchError(
                "DocxDocumentParser only accepts docx artifacts"
            )

        document = self._open_document(artifact.path)
        fragments = self._extract_fragments(document)
        if not fragments:
            raise NoExtractableContentError(
                "DOCX document contains no extractable text"
            )

        return ParsedDocument(
            artifact=artifact,
            fragments=tuple(fragments),
        )

    @staticmethod
    def _open_document(path: Path) -> DocxDocument:
        if not path.is_file():
            raise DocumentReadError(
                f"DOCX document could not be read: {path}"
            )
        try:
            return Document(path)
        except (PackageNotFoundError, BadZipFile, KeyError) as error:
            raise InvalidDocumentPackageError(
                "DOCX file is not a valid Office Open XML package"
            ) from error
        except OSError as error:
            raise DocumentReadError(
                f"DOCX document could not be read: {path}"
            ) from error

    def _extract_fragments(
        self,
        document: DocxDocument,
    ) -> list[ParsedFragment]:
        fragments: list[ParsedFragment] = []
        heading_path: list[str] = []
        table_number = 0

        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = self._normalize_text(block.text)
                if not text:
                    continue
                heading_level = self._heading_level(block)
                if heading_level is not None:
                    heading_path = heading_path[: heading_level - 1]
                    heading_path.append(text)
                self._append_fragment(
                    fragments,
                    text,
                    heading_path=heading_path,
                )
            elif isinstance(block, Table):
                table_number += 1
                for row_number, row in enumerate(
                    block.rows,
                    start=1,
                ):
                    row_text = " | ".join(
                        text
                        for cell in row.cells
                        if (text := self._normalize_text(cell.text))
                    )
                    if not row_text:
                        continue
                    self._append_fragment(
                        fragments,
                        row_text,
                        heading_path=heading_path,
                        table_number=table_number,
                        table_row_number=row_number,
                    )

        return fragments

    @staticmethod
    def _heading_level(paragraph: Paragraph) -> int | None:
        style = paragraph.style
        candidates = (style.name, style.style_id)
        for candidate in candidates:
            if not candidate:
                continue
            match = _HEADING_STYLE_PATTERN.fullmatch(candidate.strip())
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _append_fragment(
        fragments: list[ParsedFragment],
        text: str,
        *,
        heading_path: list[str],
        table_number: int | None = None,
        table_row_number: int | None = None,
    ) -> None:
        fragments.append(
            ParsedFragment(
                ordinal=len(fragments),
                text=text,
                location=EvidenceLocation(
                    heading_path=tuple(heading_path),
                    table_number=table_number,
                    table_row_number=table_row_number,
                ),
            )
        )
