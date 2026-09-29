from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain.documents.document_ocr import (
    NormalizedBoundingBox,
    OcrPageResult,
    OcrTextRegion,
    RenderedPdfPage,
)


class PaddleOcrPredictionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PaddleOcrRawRegion:
    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]


class PaddleOcrPredictor(Protocol):
    @property
    def identity(self) -> str:
        ...

    def predict(
        self,
        png_bytes: bytes,
    ) -> tuple[PaddleOcrRawRegion, ...]:
        ...


class PaddleOcrTextEngine:
    def __init__(
        self,
        predictor: PaddleOcrPredictor,
        *,
        minimum_confidence: float = 0.5,
    ) -> None:
        if minimum_confidence < 0 or minimum_confidence > 1:
            raise ValueError("minimum OCR confidence must be in [0, 1]")
        if not predictor.identity.strip():
            raise ValueError("OCR predictor identity must not be blank")
        self._predictor = predictor
        self._minimum_confidence = minimum_confidence

    @property
    def identity(self) -> str:
        return self._predictor.identity

    def recognize(self, page: RenderedPdfPage) -> OcrPageResult:
        raw_regions = self._predictor.predict(page.png_bytes)
        regions = tuple(
            region
            for raw_region in raw_regions
            if (region := self._convert_region(raw_region, page)) is not None
        )
        return OcrPageResult(
            page_number=page.page_number,
            engine_identity=self.identity,
            regions=regions,
        )

    def _convert_region(
        self,
        raw_region: PaddleOcrRawRegion,
        page: RenderedPdfPage,
    ) -> OcrTextRegion | None:
        if not raw_region.text.strip():
            return None
        if raw_region.confidence < self._minimum_confidence:
            return None
        if raw_region.confidence > 1:
            raise PaddleOcrPredictionError(
                "OCR predictor returned confidence outside [0, 1]"
            )
        if len(raw_region.polygon) < 4:
            raise PaddleOcrPredictionError(
                "OCR predictor returned an invalid text polygon"
            )

        x_coordinates = tuple(point[0] for point in raw_region.polygon)
        y_coordinates = tuple(point[1] for point in raw_region.polygon)
        left = min(x_coordinates) / page.width_pixels
        top = min(y_coordinates) / page.height_pixels
        right = max(x_coordinates) / page.width_pixels
        bottom = max(y_coordinates) / page.height_pixels
        try:
            bounding_box = NormalizedBoundingBox(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
        except ValueError as error:
            raise PaddleOcrPredictionError(
                "OCR predictor returned geometry outside the page"
            ) from error
        return OcrTextRegion(
            text=raw_region.text.strip(),
            confidence=raw_region.confidence,
            bounding_box=bounding_box,
        )
