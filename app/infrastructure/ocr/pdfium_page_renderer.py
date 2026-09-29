from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from app.domain.documents.document_ocr import RenderedPdfPage


class PdfPageRendererConfigurationError(ValueError):
    pass


class PdfPageRenderingError(RuntimeError):
    pass


class PdfPageRenderingLimitError(PdfPageRenderingError):
    pass


@dataclass(frozen=True, slots=True)
class PdfiumPageRendererConfig:
    dpi: int = 300
    max_pages: int = 100
    max_pixels_per_page: int = 40_000_000

    def __post_init__(self) -> None:
        if self.dpi < 72 or self.dpi > 600:
            raise PdfPageRendererConfigurationError(
                "PDF render DPI must be between 72 and 600"
            )
        if self.max_pages < 1:
            raise PdfPageRendererConfigurationError(
                "PDF render max_pages must be positive"
            )
        if self.max_pixels_per_page < 1:
            raise PdfPageRendererConfigurationError(
                "PDF render pixel budget must be positive"
            )


class PdfiumPageRenderer:
    def __init__(
        self,
        document_factory: Callable[[Path], Any],
        config: PdfiumPageRendererConfig = PdfiumPageRendererConfig(),
    ) -> None:
        self._document_factory = document_factory
        self._config = config

    def render(self, path: Path) -> tuple[RenderedPdfPage, ...]:
        if not path.is_file():
            raise PdfPageRenderingError(
                f"PDF document could not be rendered: {path}"
            )
        document = self._document_factory(path)
        try:
            page_count = len(document)
            if page_count > self._config.max_pages:
                raise PdfPageRenderingLimitError(
                    "PDF page count exceeds the configured render limit"
                )
            return tuple(
                self._render_page(document[page_index], page_index + 1)
                for page_index in range(page_count)
            )
        finally:
            _close_if_supported(document)

    def _render_page(
        self,
        page: Any,
        page_number: int,
    ) -> RenderedPdfPage:
        try:
            width_points, height_points = page.get_size()
            scale = self._config.dpi / 72
            projected_width = math.ceil(width_points * scale)
            projected_height = math.ceil(height_points * scale)
            if (
                projected_width * projected_height
                > self._config.max_pixels_per_page
            ):
                raise PdfPageRenderingLimitError(
                    "PDF page exceeds the configured pixel budget"
                )

            bitmap = page.render(scale=scale, rotation=0)
            try:
                image = bitmap.to_pil().convert("RGB")
                try:
                    output = BytesIO()
                    image.save(output, format="PNG")
                    return RenderedPdfPage(
                        page_number=page_number,
                        width_pixels=image.width,
                        height_pixels=image.height,
                        render_dpi=self._config.dpi,
                        png_bytes=output.getvalue(),
                    )
                finally:
                    image.close()
            finally:
                _close_if_supported(bitmap)
        finally:
            _close_if_supported(page)


def build_pdfium_page_renderer(
    config: PdfiumPageRendererConfig = PdfiumPageRendererConfig(),
) -> PdfiumPageRenderer:
    try:
        import pypdfium2
    except ImportError as error:
        raise PdfPageRendererConfigurationError(
            "install the 'ocr' optional dependencies before rendering PDFs"
        ) from error
    return PdfiumPageRenderer(pypdfium2.PdfDocument, config)


def _close_if_supported(resource: object) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()
