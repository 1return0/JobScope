from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from app.infrastructure.ocr.paddle_ocr_engine import (
    PaddleOcrPredictionError,
    PaddleOcrRawRegion,
)


class PaddleOcrDependencyError(RuntimeError):
    pass


class PaddleOcrPipelinePredictor:
    def __init__(
        self,
        pipeline: Any,
        *,
        model_identity: str,
        image_array_factory: Callable[[Image.Image], Any],
    ) -> None:
        if not model_identity.strip():
            raise ValueError("PaddleOCR model identity must not be blank")
        self._pipeline = pipeline
        self._identity = model_identity
        self._image_array_factory = image_array_factory

    @property
    def identity(self) -> str:
        return self._identity

    def predict(
        self,
        png_bytes: bytes,
    ) -> tuple[PaddleOcrRawRegion, ...]:
        image = self._decode_png(png_bytes)
        try:
            image_array = self._image_array_factory(image)
            results = tuple(self._pipeline.predict(input=image_array))
        except Exception as error:
            raise PaddleOcrPredictionError(
                "PaddleOCR inference failed"
            ) from error
        finally:
            image.close()

        if len(results) != 1:
            raise PaddleOcrPredictionError(
                "PaddleOCR must return exactly one result per page"
            )
        payload = _extract_result_payload(results[0])
        return _decode_raw_regions(payload)

    @staticmethod
    def _decode_png(png_bytes: bytes) -> Image.Image:
        try:
            with Image.open(BytesIO(png_bytes)) as source:
                source.load()
                return source.convert("RGB")
        except (OSError, ValueError) as error:
            raise PaddleOcrPredictionError(
                "rendered PDF page is not a readable PNG image"
            ) from error


def build_paddle_ocr_pipeline_predictor(
    *,
    engine: str = "transformers",
    model_identity: str = "paddleocr;pipeline=ocr;version=3.7",
    cache_folder: Path | None = None,
) -> PaddleOcrPipelinePredictor:
    if cache_folder is not None:
        cache_folder.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(
            "PADDLE_PDX_CACHE_HOME",
            str(cache_folder.resolve()),
        )
    try:
        import numpy
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise PaddleOcrDependencyError(
            "install the 'ocr' optional dependencies and one inference engine"
        ) from error

    pipeline = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine=engine,
    )
    return PaddleOcrPipelinePredictor(
        pipeline,
        model_identity=model_identity,
        image_array_factory=numpy.asarray,
    )


def _extract_result_payload(result: object) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise PaddleOcrPredictionError(
            "PaddleOCR result does not expose a JSON object"
        )
    nested = payload.get("res", payload)
    if not isinstance(nested, dict):
        raise PaddleOcrPredictionError(
            "PaddleOCR result payload is invalid"
        )
    return nested


def _decode_raw_regions(
    payload: dict[str, Any],
) -> tuple[PaddleOcrRawRegion, ...]:
    texts = payload.get("rec_texts")
    scores = payload.get("rec_scores")
    polygons = payload.get("rec_polys")
    if not all(
        isinstance(items, (list, tuple))
        for items in (texts, scores, polygons)
    ):
        raise PaddleOcrPredictionError(
            "PaddleOCR result is missing recognition arrays"
        )
    if not (len(texts) == len(scores) == len(polygons)):
        raise PaddleOcrPredictionError(
            "PaddleOCR recognition arrays have different lengths"
        )

    regions: list[PaddleOcrRawRegion] = []
    try:
        for text, score, polygon in zip(
            texts,
            scores,
            polygons,
            strict=True,
        ):
            if not isinstance(text, str):
                raise TypeError
            converted_polygon = tuple(
                (float(point[0]), float(point[1]))
                for point in polygon
            )
            regions.append(
                PaddleOcrRawRegion(
                    text=text,
                    confidence=float(score),
                    polygon=converted_polygon,
                )
            )
    except (TypeError, ValueError, IndexError) as error:
        raise PaddleOcrPredictionError(
            "PaddleOCR recognition values are invalid"
        ) from error
    return tuple(regions)
