from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from app.domain.documents.document_ingestion import (
    EvidenceLocation,
    ParsedDocument,
    ParsedFragment,
)


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    evidence_id: str
    document_sha256: str
    source_reference: str
    source_fragment_ordinal: int
    chunk_ordinal: int
    chunker_version: str
    text: str
    location: EvidenceLocation

    def __post_init__(self) -> None:
        if not self.evidence_id.startswith("ev_"):
            raise ValueError("evidence_id must start with ev_")
        if len(self.evidence_id) != 67:
            raise ValueError(
                "evidence_id must contain a full SHA-256 digest"
            )
        try:
            int(self.evidence_id.removeprefix("ev_"), 16)
        except ValueError as error:
            raise ValueError(
                "evidence_id must contain a SHA-256 hex digest"
            ) from error
        if self.source_fragment_ordinal < 0:
            raise ValueError(
                "source_fragment_ordinal must not be negative"
            )
        if self.chunk_ordinal < 0:
            raise ValueError("chunk_ordinal must not be negative")
        if not self.chunker_version.strip():
            raise ValueError("chunker_version must not be blank")
        if not self.text.strip():
            raise ValueError("chunk text must not be blank")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_chars: int = 800
    overlap_chars: int = 100
    algorithm_version: str = "structure-v1"

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError(
                "overlap_chars must be smaller than max_chars"
            )
        if not self.algorithm_version.strip():
            raise ValueError("algorithm_version must not be blank")

    @property
    def identity(self) -> str:
        return (
            f"{self.algorithm_version.strip()};"
            f"max_chars={self.max_chars};"
            f"overlap_chars={self.overlap_chars}"
        )


class StructureAwareChunker:
    _SENTENCE_ENDINGS = frozenset("。！？!?；;")

    def __init__(
        self,
        config: ChunkingConfig = ChunkingConfig(),
    ) -> None:
        self._config = config

    def chunk(
        self,
        parsed_document: ParsedDocument,
    ) -> tuple[EvidenceChunk, ...]:
        chunks: list[EvidenceChunk] = []

        for source_fragment in parsed_document.fragments:
            pieces = self._split_text(source_fragment.text)
            for chunk_ordinal, text in enumerate(pieces):
                chunks.append(
                    create_evidence_chunk(
                        parsed_document,
                        source_fragment=source_fragment,
                        chunk_ordinal=chunk_ordinal,
                        text=text,
                        chunker_version=self._config.identity,
                    )
                )

        return tuple(chunks)

    def _split_text(self, text: str) -> tuple[str, ...]:
        normalized_text = " ".join(text.split())
        if len(normalized_text) <= self._config.max_chars:
            return (normalized_text,)

        pieces: list[str] = []
        start = 0
        text_length = len(normalized_text)

        while start < text_length:
            hard_end = min(
                start + self._config.max_chars,
                text_length,
            )
            end = self._choose_end(
                normalized_text,
                start=start,
                hard_end=hard_end,
            )
            piece = normalized_text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= text_length:
                break

            next_start = max(
                start + 1,
                end - self._config.overlap_chars,
            )
            while (
                next_start < end
                and normalized_text[next_start].isspace()
            ):
                next_start += 1
            start = next_start

        return tuple(pieces)

    def _choose_end(
        self,
        text: str,
        *,
        start: int,
        hard_end: int,
    ) -> int:
        if hard_end >= len(text):
            return len(text)

        preferred_start = start + max(
            1,
            self._config.max_chars // 2,
        )
        for index in range(
            hard_end - 1,
            preferred_start - 1,
            -1,
        ):
            if text[index] in self._SENTENCE_ENDINGS:
                return index + 1

        for index in range(
            hard_end - 1,
            preferred_start - 1,
            -1,
        ):
            if text[index].isspace():
                return index

        return hard_end


def create_evidence_chunk(
    parsed_document: ParsedDocument,
    *,
    source_fragment: ParsedFragment,
    chunk_ordinal: int,
    text: str,
    chunker_version: str,
) -> EvidenceChunk:
    if not any(
        candidate is source_fragment
        for candidate in parsed_document.fragments
    ):
        raise ValueError(
            "source_fragment must belong to parsed_document"
        )
    normalized_version = chunker_version.strip()
    normalized_text = " ".join(text.split())
    evidence_id = build_evidence_id(
        document_sha256=parsed_document.artifact.content_sha256,
        source_reference=parsed_document.artifact.source_reference,
        source_fragment_ordinal=source_fragment.ordinal,
        chunk_ordinal=chunk_ordinal,
        chunker_version=normalized_version,
        text=normalized_text,
        location=source_fragment.location,
    )
    return EvidenceChunk(
        evidence_id=evidence_id,
        document_sha256=parsed_document.artifact.content_sha256,
        source_reference=parsed_document.artifact.source_reference,
        source_fragment_ordinal=source_fragment.ordinal,
        chunk_ordinal=chunk_ordinal,
        chunker_version=normalized_version,
        text=normalized_text,
        location=source_fragment.location,
    )


def build_evidence_id(
    *,
    document_sha256: str,
    source_reference: str,
    source_fragment_ordinal: int,
    chunk_ordinal: int,
    chunker_version: str,
    text: str,
    location: EvidenceLocation,
) -> str:
    identity_payload = {
        "chunk_ordinal": chunk_ordinal,
        "chunker_version": chunker_version,
        "document_sha256": document_sha256,
        "location": {
            "cell_reference": location.cell_reference,
            "heading_path": list(location.heading_path),
            "page_number": location.page_number,
            "sheet_name": location.sheet_name,
            "slide_number": location.slide_number,
            "table_number": location.table_number,
            "table_row_number": location.table_row_number,
        },
        "source_fragment_ordinal": source_fragment_ordinal,
        "source_reference": source_reference,
        "text": text,
    }
    canonical_json = json.dumps(
        identity_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return f"ev_{digest}"
