from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    EvidenceLocation,
    NoExtractableContentError,
    ParsedDocument,
    ParsedFragment,
)
from app.domain.documents.document_ocr import OcrPageResult, OcrTextRegion


@dataclass(slots=True)
class _OcrTextLine:
    regions: list[OcrTextRegion] = field(default_factory=list)
    top: float = 1.0
    bottom: float = 0.0

    def vertical_overlap_ratio(self, region: OcrTextRegion) -> float:
        box = region.bounding_box
        overlap = min(self.bottom, box.bottom) - max(self.top, box.top)
        if overlap <= 0:
            return 0.0
        line_height = self.bottom - self.top
        region_height = box.bottom - box.top
        return overlap / min(line_height, region_height)

    def add(self, region: OcrTextRegion) -> None:
        self.regions.append(region)
        self.top = min(self.top, region.bounding_box.top)
        self.bottom = max(self.bottom, region.bounding_box.bottom)


class SingleColumnOcrReadingOrder:
    def __init__(self, *, minimum_line_overlap: float = 0.5) -> None:
        if minimum_line_overlap <= 0 or minimum_line_overlap > 1:
            raise ValueError("minimum_line_overlap must be in (0, 1]")
        self._minimum_line_overlap = minimum_line_overlap

    def order(
        self,
        regions: tuple[OcrTextRegion, ...],
    ) -> tuple[OcrTextRegion, ...]:
        lines: list[_OcrTextLine] = []
        for region in sorted(
            regions,
            key=lambda item: (
                item.bounding_box.top,
                item.bounding_box.left,
            ),
        ):
            matching_line = self._find_matching_line(lines, region)
            if matching_line is None:
                matching_line = _OcrTextLine()
                lines.append(matching_line)
            matching_line.add(region)

        ordered: list[OcrTextRegion] = []
        for line in sorted(lines, key=lambda item: item.top):
            ordered.extend(
                sorted(
                    line.regions,
                    key=lambda item: item.bounding_box.left,
                )
            )
        return tuple(ordered)

    def _find_matching_line(
        self,
        lines: list[_OcrTextLine],
        region: OcrTextRegion,
    ) -> _OcrTextLine | None:
        candidates = tuple(
            (line.vertical_overlap_ratio(region), line)
            for line in lines
        )
        if not candidates:
            return None
        overlap, line = max(candidates, key=lambda item: item[0])
        if overlap < self._minimum_line_overlap:
            return None
        return line


class OcrParsedDocumentAssembler:
    def __init__(
        self,
        reading_order: SingleColumnOcrReadingOrder,
    ) -> None:
        self._reading_order = reading_order

    def assemble(
        self,
        artifact: DocumentArtifact,
        pages: tuple[OcrPageResult, ...],
    ) -> ParsedDocument:
        fragments: list[ParsedFragment] = []
        for page in sorted(pages, key=lambda item: item.page_number):
            ordered_regions = self._reading_order.order(page.regions)
            if not ordered_regions:
                continue
            fragments.append(
                ParsedFragment(
                    ordinal=len(fragments),
                    text="\n".join(
                        region.text for region in ordered_regions
                    ),
                    location=EvidenceLocation(
                        page_number=page.page_number,
                    ),
                )
            )

        if not fragments:
            raise NoExtractableContentError(
                "OCR produced no extractable text"
            )
        return ParsedDocument(
            artifact=artifact,
            fragments=tuple(fragments),
        )
