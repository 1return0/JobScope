import unittest

from app.domain.documents.document_ocr import RenderedPdfPage
from app.infrastructure.ocr.paddle_ocr_engine import (
    PaddleOcrPredictionError,
    PaddleOcrRawRegion,
    PaddleOcrTextEngine,
)


class _FakePaddleOcrPredictor:
    identity = "paddleocr;model=PP-OCRv5_mobile;revision=test"

    def __init__(
        self,
        regions: tuple[PaddleOcrRawRegion, ...],
    ) -> None:
        self._regions = regions
        self.received_png_bytes: bytes | None = None

    def predict(
        self,
        png_bytes: bytes,
    ) -> tuple[PaddleOcrRawRegion, ...]:
        self.received_png_bytes = png_bytes
        return self._regions


def _page() -> RenderedPdfPage:
    return RenderedPdfPage(
        page_number=3,
        width_pixels=1000,
        height_pixels=2000,
        render_dpi=300,
        png_bytes=b"test-png",
    )


class PaddleOcrTextEngineTest(unittest.TestCase):
    def test_converts_pixel_polygon_to_normalized_domain_region(self) -> None:
        predictor = _FakePaddleOcrPredictor(
            (
                PaddleOcrRawRegion(
                    text="  专业不限  ",
                    confidence=0.96,
                    polygon=(
                        (100, 400),
                        (500, 400),
                        (500, 500),
                        (100, 500),
                    ),
                ),
            )
        )
        engine = PaddleOcrTextEngine(
            predictor,
            minimum_confidence=0.8,
        )

        result = engine.recognize(_page())

        self.assertEqual(b"test-png", predictor.received_png_bytes)
        self.assertEqual(3, result.page_number)
        self.assertEqual(predictor.identity, result.engine_identity)
        self.assertEqual("专业不限", result.regions[0].text)
        self.assertEqual(0.1, result.regions[0].bounding_box.left)
        self.assertEqual(0.2, result.regions[0].bounding_box.top)
        self.assertEqual(0.5, result.regions[0].bounding_box.right)
        self.assertEqual(0.25, result.regions[0].bounding_box.bottom)

    def test_filters_blank_and_low_confidence_regions(self) -> None:
        predictor = _FakePaddleOcrPredictor(
            (
                PaddleOcrRawRegion(
                    text="",
                    confidence=0.99,
                    polygon=((0, 0), (10, 0), (10, 10), (0, 10)),
                ),
                PaddleOcrRawRegion(
                    text="模糊文字",
                    confidence=0.49,
                    polygon=((0, 0), (10, 0), (10, 10), (0, 10)),
                ),
            )
        )

        result = PaddleOcrTextEngine(predictor).recognize(_page())

        self.assertEqual((), result.regions)

    def test_rejects_geometry_outside_rendered_page(self) -> None:
        predictor = _FakePaddleOcrPredictor(
            (
                PaddleOcrRawRegion(
                    text="越界",
                    confidence=0.9,
                    polygon=(
                        (900, 100),
                        (1100, 100),
                        (1100, 200),
                        (900, 200),
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(
            PaddleOcrPredictionError,
            "outside the page",
        ):
            PaddleOcrTextEngine(predictor).recognize(_page())


if __name__ == "__main__":
    unittest.main()
