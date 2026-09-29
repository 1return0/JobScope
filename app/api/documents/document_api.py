from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel

from app.application.documents.document_parsing import (
    DocumentParsingResult,
    DocumentParsingService,
)
from app.application.documents.document_processing import (
    DocumentProcessingResult,
    DocumentProcessingService,
)
from app.domain.documents.document_ingestion import (
    DocumentFormat,
    EvidenceLocation,
    UnsupportedDocumentFormatError,
    detect_document_format,
    inspect_document_artifact,
)


class EvidenceLocationResponse(BaseModel):
    page_number: int | None
    heading_path: list[str]
    sheet_name: str | None
    cell_reference: str | None
    slide_number: int | None
    table_number: int | None
    table_row_number: int | None


class ParsedFragmentResponse(BaseModel):
    ordinal: int
    text: str
    location: EvidenceLocationResponse


class ParsingFailureResponse(BaseModel):
    code: str
    message: str
    retryable: bool
    operator_action: str


class DocumentParseResponse(BaseModel):
    status: Literal["succeeded", "failed"]
    filename: str
    document_format: DocumentFormat
    content_sha256: str
    byte_size: int
    fragments: list[ParsedFragmentResponse]
    failure: ParsingFailureResponse | None


class EvidenceChunkResponse(BaseModel):
    evidence_id: str
    source_fragment_ordinal: int
    chunk_ordinal: int
    chunker_version: str
    text: str
    location: EvidenceLocationResponse


class DocumentProcessResponse(BaseModel):
    status: Literal["succeeded", "failed"]
    filename: str
    document_format: DocumentFormat
    content_sha256: str
    byte_size: int
    fragment_count: int
    chunk_count: int
    chunks: list[EvidenceChunkResponse]
    failure: ParsingFailureResponse | None


def build_document_router(
    parsing_service: DocumentParsingService,
    processing_service: DocumentProcessingService,
    *,
    max_upload_bytes: int,
) -> APIRouter:
    if max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive")

    router = APIRouter(prefix="/v1/documents", tags=["documents"])

    @router.post(
        "/parse",
        response_model=DocumentParseResponse,
        summary="Parse a supported recruitment document",
        description=(
            "Temporarily stages one uploaded document, records its content "
            "identity, and parses it without persisting source content."
        ),
    )
    async def parse_document(
        file: Annotated[UploadFile, File(...)],
        source_reference: Annotated[
            str,
            Form(min_length=1, max_length=2048),
        ],
    ) -> DocumentParseResponse:
        safe_filename = _validate_filename(file.filename)
        document_format = _validate_extension(safe_filename)
        normalized_source_reference = source_reference.strip()
        if not normalized_source_reference:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="source_reference must not be blank",
            )

        try:
            with TemporaryDirectory(
                prefix="jobscope-upload-"
            ) as temporary_directory:
                path = Path(temporary_directory) / safe_filename
                await _stage_upload(
                    file,
                    path,
                    max_upload_bytes=max_upload_bytes,
                )
                artifact = inspect_document_artifact(
                    path,
                    source_reference=normalized_source_reference,
                )
                if artifact.document_format != document_format:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="uploaded file format could not be confirmed",
                    )
                result = parsing_service.parse(artifact)
                return _build_response(
                    result,
                    filename=safe_filename,
                )
        finally:
            await file.close()

    @router.post(
        "/process",
        response_model=DocumentProcessResponse,
        summary="Parse and chunk a supported recruitment document",
        description=(
            "Temporarily stages, parses, and structure-aware chunks one "
            "document for inspection. It does not persist or index content."
        ),
    )
    async def process_document(
        file: Annotated[UploadFile, File(...)],
        source_reference: Annotated[
            str,
            Form(min_length=1, max_length=2048),
        ],
    ) -> DocumentProcessResponse:
        safe_filename = _validate_filename(file.filename)
        document_format = _validate_extension(safe_filename)
        normalized_source_reference = source_reference.strip()
        if not normalized_source_reference:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="source_reference must not be blank",
            )

        try:
            with TemporaryDirectory(
                prefix="jobscope-upload-"
            ) as temporary_directory:
                path = Path(temporary_directory) / safe_filename
                await _stage_upload(
                    file,
                    path,
                    max_upload_bytes=max_upload_bytes,
                )
                artifact = inspect_document_artifact(
                    path,
                    source_reference=normalized_source_reference,
                )
                if artifact.document_format != document_format:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="uploaded file format could not be confirmed",
                    )
                result = processing_service.process(artifact)
                return _build_process_response(
                    result,
                    filename=safe_filename,
                )
        finally:
            await file.close()

    return router


def _validate_filename(filename: str | None) -> str:
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file must have a filename",
        )
    safe_filename = Path(filename).name.strip()
    if not safe_filename or safe_filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file must have a valid filename",
        )
    return safe_filename


def _validate_extension(filename: str) -> DocumentFormat:
    try:
        return detect_document_format(Path(filename))
    except UnsupportedDocumentFormatError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="unsupported document extension",
        ) from error


async def _stage_upload(
    upload: UploadFile,
    path: Path,
    *,
    max_upload_bytes: int,
) -> None:
    byte_size = 0
    with path.open("wb") as staged_file:
        while chunk := await upload.read(1024 * 1024):
            byte_size += len(chunk)
            if byte_size > max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=(
                        "uploaded file exceeds the configured size limit"
                    ),
                )
            staged_file.write(chunk)


def _build_response(
    result: DocumentParsingResult,
    *,
    filename: str,
) -> DocumentParseResponse:
    artifact = result.artifact
    if result.succeeded:
        assert result.parsed_document is not None
        return DocumentParseResponse(
            status="succeeded",
            filename=filename,
            document_format=artifact.document_format,
            content_sha256=artifact.content_sha256,
            byte_size=artifact.byte_size,
            fragments=[
                ParsedFragmentResponse(
                    ordinal=fragment.ordinal,
                    text=fragment.text,
                    location=EvidenceLocationResponse(
                        page_number=fragment.location.page_number,
                        heading_path=list(
                            fragment.location.heading_path
                        ),
                        sheet_name=fragment.location.sheet_name,
                        cell_reference=(
                            fragment.location.cell_reference
                        ),
                        slide_number=fragment.location.slide_number,
                        table_number=fragment.location.table_number,
                        table_row_number=(
                            fragment.location.table_row_number
                        ),
                    ),
                )
                for fragment in result.parsed_document.fragments
            ],
            failure=None,
        )

    assert result.failure is not None
    return DocumentParseResponse(
        status="failed",
        filename=filename,
        document_format=artifact.document_format,
        content_sha256=artifact.content_sha256,
        byte_size=artifact.byte_size,
        fragments=[],
        failure=ParsingFailureResponse(
            code=result.failure.code,
            message=result.failure.message,
            retryable=result.failure.retryable,
            operator_action=result.failure.operator_action,
        ),
    )


def _location_response(
    location: EvidenceLocation,
) -> EvidenceLocationResponse:
    return EvidenceLocationResponse(
        page_number=location.page_number,
        heading_path=list(location.heading_path),
        sheet_name=location.sheet_name,
        cell_reference=location.cell_reference,
        slide_number=location.slide_number,
        table_number=location.table_number,
        table_row_number=location.table_row_number,
    )


def _build_process_response(
    result: DocumentProcessingResult,
    *,
    filename: str,
) -> DocumentProcessResponse:
    artifact = result.artifact
    parsing_result = result.parsing_result
    if result.succeeded:
        parsed_document = parsing_result.parsed_document
        assert parsed_document is not None
        return DocumentProcessResponse(
            status="succeeded",
            filename=filename,
            document_format=artifact.document_format,
            content_sha256=artifact.content_sha256,
            byte_size=artifact.byte_size,
            fragment_count=len(parsed_document.fragments),
            chunk_count=len(result.chunks),
            chunks=[
                EvidenceChunkResponse(
                    evidence_id=chunk.evidence_id,
                    source_fragment_ordinal=(
                        chunk.source_fragment_ordinal
                    ),
                    chunk_ordinal=chunk.chunk_ordinal,
                    chunker_version=chunk.chunker_version,
                    text=chunk.text,
                    location=_location_response(chunk.location),
                )
                for chunk in result.chunks
            ],
            failure=None,
        )

    failure = parsing_result.failure
    assert failure is not None
    return DocumentProcessResponse(
        status="failed",
        filename=filename,
        document_format=artifact.document_format,
        content_sha256=artifact.content_sha256,
        byte_size=artifact.byte_size,
        fragment_count=0,
        chunk_count=0,
        chunks=[],
        failure=ParsingFailureResponse(
            code=failure.code,
            message=failure.message,
            retryable=failure.retryable,
            operator_action=failure.operator_action,
        ),
    )
