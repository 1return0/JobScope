import unittest
from pathlib import Path

from app.application.ocr.pdf_ocr_parsing import (
    OcrParsedDocumentAssembler,
    SingleColumnOcrReadingOrder,
)
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    NoExtractableContentError,
)
from app.domain.documents.document_ocr import (
    NormalizedBoundingBox,
    OcrPageResult,
    OcrTextRegion,
)


def _region(
    text: str,
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> OcrTextRegion:
    return OcrTextRegion(
        text=text,
        confidence=0.95,
        bounding_box=NormalizedBoundingBox(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        ),
    )


def _artifact() -> DocumentArtifact:
    return DocumentArtifact(
        path=Path("scan.pdf"),
        source_reference="https://example.edu/jobs.pdf",
        document_format="pdf",
        content_sha256="a" * 64,
        byte_size=100,
    )


class SingleColumnOcrReadingOrderTest(unittest.TestCase):
    def test_orders_lines_top_to_bottom_and_regions_left_to_right(self) -> None:
        order = SingleColumnOcrReadingOrder()
        regions = (
            _region(
                "开发工程师",
                left=0.4,
                top=0.1,
                right=0.8,
                bottom=0.15,
            ),
            _region(
                "岗位：",
                left=0.1,
                top=0.105,
                right=0.3,
                bottom=0.155,
            ),
            _region(
                "专业不限",
                left=0.1,
                top=0.3,
                right=0.4,
                bottom=0.35,
            ),
        )

        result = order.order(regions)

        self.assertEqual(
            ("岗位：", "开发工程师", "专业不限"),
            tuple(region.text for region in result),
        )


class OcrParsedDocumentAssemblerTest(unittest.TestCase):
    def test_builds_one_fragment_per_non_empty_page(self) -> None:
        assembler = OcrParsedDocumentAssembler(
            SingleColumnOcrReadingOrder()
        )
        pages = (
            OcrPageResult(
                page_number=2,
                engine_identity="test-ocr;revision=1",
                regions=(
                    _region(
                        "第二页",
                        left=0.1,
                        top=0.1,
                        right=0.3,
                        bottom=0.2,
                    ),
                ),
            ),
            OcrPageResult(
                page_number=1,
                engine_identity="test-ocr;revision=1",
                regions=(
                    _region(
                        "专业不限",
                        left=0.1,
                        top=0.2,
                        right=0.4,
                        bottom=0.3,
                    ),
                    _region(
                        "招聘要求",
                        left=0.1,
                        top=0.1,
                        right=0.4,
                        bottom=0.15,
                    ),
                ),
            ),
        )

        result = assembler.assemble(_artifact(), pages)

        self.assertEqual((0, 1), tuple(f.ordinal for f in result.fragments))
        self.assertEqual(
            (1, 2),
            tuple(f.location.page_number for f in result.fragments),
        )
        self.assertEqual(
            "招聘要求\n专业不限",
            result.fragments[0].text,
        )

    def test_rejects_an_all_empty_ocr_result(self) -> None:
        assembler = OcrParsedDocumentAssembler(
            SingleColumnOcrReadingOrder()
        )
        pages = (
            OcrPageResult(
                page_number=1,
                engine_identity="test-ocr;revision=1",
                regions=(),
            ),
        )

        with self.assertRaisesRegex(
            NoExtractableContentError,
            "OCR produced no extractable text",
        ):
            assembler.assemble(_artifact(), pages)


if __name__ == "__main__":
    unittest.main()
