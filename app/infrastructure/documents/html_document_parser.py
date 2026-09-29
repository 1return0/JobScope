from __future__ import annotations

from html.parser import HTMLParser

from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    DocumentFormat,
    DocumentFormatMismatchError,
    DocumentReadError,
    EvidenceLocation,
    NoExtractableContentError,
    ParsedDocument,
    ParsedFragment,
    UnsupportedTextEncodingError,
)


class HtmlDocumentParser:
    document_format: DocumentFormat = "html"

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        if artifact.document_format != self.document_format:
            raise DocumentFormatMismatchError(
                "HtmlDocumentParser only accepts html artifacts"
            )

        try:
            html = artifact.path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise UnsupportedTextEncodingError(
                "HTML document is not valid UTF-8 text"
            ) from error
        except OSError as error:
            raise DocumentReadError(
                f"HTML document could not be read: {artifact.path}"
            ) from error

        extractor = _StructuredHtmlTextExtractor()
        extractor.feed(html)
        extractor.close()

        if not extractor.fragments:
            raise NoExtractableContentError(
                "HTML document contains no supported textual content"
            )

        return ParsedDocument(
            artifact=artifact,
            fragments=tuple(extractor.fragments),
        )


class _StructuredHtmlTextExtractor(HTMLParser):
    _CONTENT_TAGS = frozenset(
        {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}
    )
    _IGNORED_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[ParsedFragment] = []
        self._heading_path: list[str] = []
        self._active_tag: str | None = None
        self._text_parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.casefold()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth == 0 and tag in self._CONTENT_TAGS:
            self._flush_active_block()
            self._active_tag = tag

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth == 0 and tag == self._active_tag:
            self._flush_active_block()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0 and self._active_tag is not None:
            self._text_parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush_active_block()

    def _flush_active_block(self) -> None:
        if self._active_tag is None:
            return

        text = " ".join(" ".join(self._text_parts).split())
        tag = self._active_tag
        self._active_tag = None
        self._text_parts = []
        if not text:
            return

        if tag.startswith("h"):
            heading_level = int(tag[1])
            self._heading_path = self._heading_path[: heading_level - 1]
            self._heading_path.append(text)

        self.fragments.append(
            ParsedFragment(
                ordinal=len(self.fragments),
                text=text,
                location=EvidenceLocation(
                    heading_path=tuple(self._heading_path),
                ),
            )
        )
