import unittest

from app.domain.documents.document_ingestion import (
    DocumentReadError,
    InvalidPdfError,
    PasswordProtectedDocumentError,
    PdfTextLayerMissingError,
)
from app.domain.documents.document_ocr import (
    NormalizedBoundingBox,
    OcrPageResult,
    OcrTextRegion,
    PdfOcrFallbackPolicy,
    RenderedPdfPage,
)


class DocumentOcrContractTest(unittest.TestCase):
    def test_only_missing_text_layer_is_eligible_for_ocr_fallback(self) -> None:
        policy = PdfOcrFallbackPolicy()

        self.assertTrue(
            policy.should_attempt(PdfTextLayerMissingError("no text"))
        )
        self.assertFalse(policy.should_attempt(InvalidPdfError("invalid")))
        self.assertFalse(
            policy.should_attempt(
                PasswordProtectedDocumentError("password required")
            )
        )
        self.assertFalse(
            policy.should_attempt(DocumentReadError("read failed"))
        )

    def test_preserves_page_geometry_confidence_and_engine_identity(
        self,
    ) -> None:
        page = RenderedPdfPage(
            page_number=2,
            width_pixels=2480,
            height_pixels=3508,
            render_dpi=300,
            png_bytes=b"png-image",
        )
        region = OcrTextRegion(
            text="招聘岗位",
            confidence=0.97,
            bounding_box=NormalizedBoundingBox(
                left=0.1,
                top=0.2,
                right=0.5,
                bottom=0.25,
            ),
        )
        result = OcrPageResult(
            page_number=page.page_number,
            engine_identity="test-ocr;model=v1",
            regions=(region,),
        )

        self.assertEqual(2, result.page_number)
        self.assertEqual(0.97, result.regions[0].confidence)
        self.assertEqual(0.1, result.regions[0].bounding_box.left)
        self.assertEqual("test-ocr;model=v1", result.engine_identity)

    def test_rejects_invalid_confidence_and_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence"):
            OcrTextRegion(
                text="岗位",
                confidence=1.1,
                bounding_box=NormalizedBoundingBox(
                    left=0.1,
                    top=0.1,
                    right=0.2,
                    bottom=0.2,
                ),
            )

        with self.assertRaisesRegex(ValueError, "positive area"):
            NormalizedBoundingBox(
                left=0.5,
                top=0.1,
                right=0.5,
                bottom=0.2,
            )


if __name__ == "__main__":
    unittest.main()
