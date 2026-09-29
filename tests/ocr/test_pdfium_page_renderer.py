import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from app.infrastructure.ocr.pdfium_page_renderer import (
    PdfiumPageRenderer,
    PdfiumPageRendererConfig,
    PdfPageRenderingLimitError,
)


class _FakeBitmap:
    def __init__(self, width: int, height: int) -> None:
        self._image = Image.new("RGB", (width, height), "white")
        self.closed = False

    def to_pil(self) -> Image.Image:
        return self._image.copy()

    def close(self) -> None:
        self.closed = True
        self._image.close()


class _FakePage:
    def __init__(self, width_points: int, height_points: int) -> None:
        self._width_points = width_points
        self._height_points = height_points
        self.render_calls: list[tuple[float, int]] = []
        self.closed = False

    def get_size(self) -> tuple[int, int]:
        return self._width_points, self._height_points

    def render(self, *, scale: float, rotation: int) -> _FakeBitmap:
        self.render_calls.append((scale, rotation))
        return _FakeBitmap(
            round(self._width_points * scale),
            round(self._height_points * scale),
        )

    def close(self) -> None:
        self.closed = True


class _FakeDocument:
    def __init__(self, pages: tuple[_FakePage, ...]) -> None:
        self._pages = pages
        self.closed = False

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


class PdfiumPageRendererTest(unittest.TestCase):
    def test_renders_pages_to_png_with_original_page_numbers(self) -> None:
        first = _FakePage(72, 144)
        second = _FakePage(144, 72)
        document = _FakeDocument((first, second))
        renderer = PdfiumPageRenderer(
            lambda path: document,
            PdfiumPageRendererConfig(
                dpi=144,
                max_pages=2,
                max_pixels_per_page=1_000_000,
            ),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            path.write_bytes(b"%PDF-test")
            pages = renderer.render(path)

        self.assertEqual((1, 2), tuple(page.page_number for page in pages))
        self.assertEqual((144, 288), (pages[0].width_pixels, pages[0].height_pixels))
        self.assertTrue(pages[0].png_bytes.startswith(b"\x89PNG"))
        self.assertEqual([(2.0, 0)], first.render_calls)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertTrue(document.closed)

    def test_rejects_page_count_before_rendering(self) -> None:
        pages = (_FakePage(72, 72), _FakePage(72, 72))
        document = _FakeDocument(pages)
        renderer = PdfiumPageRenderer(
            lambda path: document,
            PdfiumPageRendererConfig(max_pages=1),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "large.pdf"
            path.write_bytes(b"%PDF-test")
            with self.assertRaisesRegex(
                PdfPageRenderingLimitError,
                "page count",
            ):
                renderer.render(path)

        self.assertEqual([], pages[0].render_calls)
        self.assertTrue(document.closed)

    def test_rejects_projected_pixels_before_allocating_bitmap(self) -> None:
        page = _FakePage(1000, 1000)
        document = _FakeDocument((page,))
        renderer = PdfiumPageRenderer(
            lambda path: document,
            PdfiumPageRendererConfig(
                dpi=300,
                max_pixels_per_page=1_000_000,
            ),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.pdf"
            path.write_bytes(b"%PDF-test")
            with self.assertRaisesRegex(
                PdfPageRenderingLimitError,
                "pixel budget",
            ):
                renderer.render(path)

        self.assertEqual([], page.render_calls)
        self.assertTrue(page.closed)
        self.assertTrue(document.closed)


if __name__ == "__main__":
    unittest.main()
