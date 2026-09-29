from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.domain.documents.document_ingestion import (
    DocumentParsingError,
    PdfTextLayerMissingError,
)


@dataclass(frozen=True, slots=True)
class RenderedPdfPage:
    page_number: int
    width_pixels: int
    height_pixels: int
    render_dpi: int
    png_bytes: bytes

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("rendered PDF page number must be positive")
        if self.width_pixels < 1 or self.height_pixels < 1:
            raise ValueError("rendered PDF page dimensions must be positive")
        if self.render_dpi < 72:
            raise ValueError("rendered PDF page DPI must be at least 72")
        if not self.png_bytes:
            raise ValueError("rendered PDF page image must not be empty")


@dataclass(frozen=True, slots=True)
class NormalizedBoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        coordinates = (self.left, self.top, self.right, self.bottom)
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError("OCR bounding-box coordinates must be in [0, 1]")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("OCR bounding box must have positive area")


@dataclass(frozen=True, slots=True)
class OcrTextRegion:
    text: str
    confidence: float
    bounding_box: NormalizedBoundingBox

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("OCR text region must not be blank")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("OCR confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    page_number: int
    engine_identity: str
    regions: tuple[OcrTextRegion, ...]

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("OCR result page number must be positive")
        if not self.engine_identity.strip():
            raise ValueError("OCR engine identity must not be blank")


class OcrEngine(Protocol):
    @property
    def identity(self) -> str:
        ...

    def recognize(self, page: RenderedPdfPage) -> OcrPageResult:
        ...


class PdfPageRenderer(Protocol):
    def render(self, path: Path) -> tuple[RenderedPdfPage, ...]:
        ...


class PdfOcrFallbackPolicy:
    def should_attempt(self, error: DocumentParsingError) -> bool:
        return isinstance(error, PdfTextLayerMissingError)
