import os
import unittest
from unittest.mock import patch

from app.config import load_settings


class PdfOcrSettingsTest(unittest.TestCase):
    def test_ocr_is_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings(dotenv_override=False)

        self.assertFalse(settings.pdf_ocr_enabled)

    def test_loads_ocr_resource_limits_from_environment(self) -> None:
        environment = {
            "JOBSCOPE_PDF_OCR_ENABLED": "true",
            "JOBSCOPE_PDF_OCR_ENGINE": "transformers",
            "JOBSCOPE_PDF_OCR_CACHE_FOLDER": ".cache-test/paddlex",
            "JOBSCOPE_PDF_OCR_MINIMUM_CONFIDENCE": "0.75",
            "JOBSCOPE_PDF_OCR_RENDER_DPI": "240",
            "JOBSCOPE_PDF_OCR_MAX_PAGES": "25",
            "JOBSCOPE_PDF_OCR_MAX_PIXELS_PER_PAGE": "12000000",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = load_settings(dotenv_override=False)

        self.assertTrue(settings.pdf_ocr_enabled)
        self.assertEqual("transformers", settings.pdf_ocr_engine)
        self.assertEqual(
            ".cache-test/paddlex",
            settings.pdf_ocr_cache_folder.as_posix(),
        )
        self.assertEqual(0.75, settings.pdf_ocr_minimum_confidence)
        self.assertEqual(240, settings.pdf_ocr_render_dpi)
        self.assertEqual(25, settings.pdf_ocr_max_pages)
        self.assertEqual(
            12_000_000,
            settings.pdf_ocr_max_pixels_per_page,
        )


if __name__ == "__main__":
    unittest.main()
