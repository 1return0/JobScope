import unittest
from io import BytesIO

from PIL import Image

from app.infrastructure.ocr.paddle_ocr_engine import PaddleOcrPredictionError
from app.infrastructure.ocr.paddle_ocr_pipeline_predictor import (
    PaddleOcrPipelinePredictor,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (20, 10), "white")
    try:
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    finally:
        image.close()


class _Result:
    def __init__(self, payload: dict[str, object]) -> None:
        self.json = payload


class _Pipeline:
    def __init__(self, results: list[_Result]) -> None:
        self._results = results
        self.received_input: object | None = None

    def predict(self, *, input: object) -> list[_Result]:
        self.received_input = input
        return self._results


class PaddleOcrPipelinePredictorTest(unittest.TestCase):
    def test_decodes_png_and_maps_official_recognition_fields(self) -> None:
        pipeline = _Pipeline(
            [
                _Result(
                    {
                        "res": {
                            "rec_texts": ["专业不限"],
                            "rec_scores": [0.97],
                            "rec_polys": [
                                [[1, 2], [9, 2], [9, 6], [1, 6]]
                            ],
                        }
                    }
                )
            ]
        )
        received_sizes: list[tuple[int, int]] = []
        predictor = PaddleOcrPipelinePredictor(
            pipeline,
            model_identity="paddleocr;pipeline=ocr;revision=test",
            image_array_factory=lambda image: received_sizes.append(
                image.size
            ) or "image-array",
        )

        regions = predictor.predict(_png_bytes())

        self.assertEqual([(20, 10)], received_sizes)
        self.assertEqual("image-array", pipeline.received_input)
        self.assertEqual("专业不限", regions[0].text)
        self.assertEqual(0.97, regions[0].confidence)
        self.assertEqual((1.0, 2.0), regions[0].polygon[0])

    def test_rejects_recognition_arrays_with_different_lengths(self) -> None:
        pipeline = _Pipeline(
            [
                _Result(
                    {
                        "rec_texts": ["A", "B"],
                        "rec_scores": [0.9],
                        "rec_polys": [
                            [[0, 0], [1, 0], [1, 1], [0, 1]]
                        ],
                    }
                )
            ]
        )
        predictor = PaddleOcrPipelinePredictor(
            pipeline,
            model_identity="test",
            image_array_factory=lambda image: "image-array",
        )

        with self.assertRaisesRegex(
            PaddleOcrPredictionError,
            "different lengths",
        ):
            predictor.predict(_png_bytes())

    def test_rejects_invalid_png_before_calling_pipeline(self) -> None:
        pipeline = _Pipeline([])
        predictor = PaddleOcrPipelinePredictor(
            pipeline,
            model_identity="test",
            image_array_factory=lambda image: "image-array",
        )

        with self.assertRaisesRegex(
            PaddleOcrPredictionError,
            "readable PNG",
        ):
            predictor.predict(b"not-png")

        self.assertIsNone(pipeline.received_input)


if __name__ == "__main__":
    unittest.main()
