from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    DocumentFormat,
    DocumentParsingError,
    ParsingFailureCode,
    ParsedDocument,
)


class DocumentParser(Protocol):
    @property
    def document_format(self) -> DocumentFormat:
        ...

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        ...


class ParserNotRegisteredError(DocumentParsingError):
    failure_code = "parser-not-registered"
    operator_action = "configure-parser"


class DuplicateParserRegistrationError(ValueError):
    pass


class InvalidParserOutputError(DocumentParsingError):
    failure_code = "invalid-parser-output"
    operator_action = "investigate-parser-contract"


@dataclass(frozen=True, slots=True)
class ParsingFailure:
    code: ParsingFailureCode
    message: str
    retryable: bool
    operator_action: str


@dataclass(frozen=True, slots=True)
class DocumentParsingResult:
    artifact: DocumentArtifact
    parsed_document: ParsedDocument | None = None
    failure: ParsingFailure | None = None

    def __post_init__(self) -> None:
        has_document = self.parsed_document is not None
        has_failure = self.failure is not None
        if has_document == has_failure:
            raise ValueError(
                "result must contain exactly one of document or failure"
            )

    @property
    def succeeded(self) -> bool:
        return self.parsed_document is not None


class DocumentParserRegistry:
    def __init__(
        self,
        parsers: Iterable[DocumentParser] = (),
    ) -> None:
        self._parsers: dict[DocumentFormat, DocumentParser] = {}
        for parser in parsers:
            self.register(parser)

    @property
    def registered_formats(self) -> frozenset[DocumentFormat]:
        return frozenset(self._parsers)

    def register(self, parser: DocumentParser) -> None:
        document_format = parser.document_format
        if document_format in self._parsers:
            raise DuplicateParserRegistrationError(
                f"parser already registered for format: {document_format}"
            )
        self._parsers[document_format] = parser

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        try:
            parser = self._parsers[artifact.document_format]
        except KeyError as error:
            raise ParserNotRegisteredError(
                "no parser registered for format: "
                f"{artifact.document_format}"
            ) from error

        parsed_document = parser.parse(artifact)
        if parsed_document.artifact != artifact:
            raise InvalidParserOutputError(
                "parser output must refer to the input artifact"
            )
        return parsed_document


class DocumentParsingService:
    def __init__(self, registry: DocumentParserRegistry) -> None:
        self._registry = registry

    def parse(
        self,
        artifact: DocumentArtifact,
    ) -> DocumentParsingResult:
        try:
            parsed_document = self._registry.parse(artifact)
        except DocumentParsingError as error:
            return DocumentParsingResult(
                artifact=artifact,
                failure=ParsingFailure(
                    code=error.failure_code,
                    message=str(error),
                    retryable=error.retryable,
                    operator_action=error.operator_action,
                ),
            )

        return DocumentParsingResult(
            artifact=artifact,
            parsed_document=parsed_document,
        )
