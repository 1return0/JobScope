from __future__ import annotations

import argparse
from html import escape
from html.parser import HTMLParser
from pathlib import Path


class BaiduJobDomExtractor(HTMLParser):
    _CLASS_TARGETS = (
        ("detail-title__", "title"),
        ("post-subtitle-item__", "metadata"),
        ("post-content-title__", "section_title"),
        ("post-content-desc__", "section_content"),
    )
    _METADATA_LABELS = (
        "招聘部门",
        "工作地点",
        "招聘项目",
        "职位类别",
        "招聘人数",
        "发布日期",
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.metadata: list[str] = []
        self.sections: list[tuple[str, str]] = []
        self._active_kind: str | None = None
        self._active_depth = 0
        self._text_parts: list[str] = []
        self._pending_section_title: str | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del tag
        if self._active_kind is not None:
            self._active_depth += 1
            return
        class_value = next(
            (value for name, value in attrs if name == "class"),
            "",
        ) or ""
        for prefix, kind in self._CLASS_TARGETS:
            if any(item.startswith(prefix) for item in class_value.split()):
                self._active_kind = kind
                self._active_depth = 1
                self._text_parts = []
                break

    def handle_endtag(self, tag: str) -> None:
        del tag
        if self._active_kind is None:
            return
        self._active_depth -= 1
        if self._active_depth == 0:
            self._flush_target()

    def handle_data(self, data: str) -> None:
        if self._active_kind is not None:
            self._text_parts.append(data)

    def _flush_target(self) -> None:
        kind = self._active_kind
        text = "\n".join(
            line.strip()
            for raw_line in self._text_parts
            for line in raw_line.splitlines()
            if line.strip()
        )
        self._active_kind = None
        self._text_parts = []
        if not text:
            return
        if kind == "title" and self.title is None:
            self.title = text
        elif kind == "metadata":
            self.metadata.append(text)
        elif kind == "section_title":
            self._pending_section_title = text.rstrip("：:")
        elif kind == "section_content":
            self.sections.append(
                (self._pending_section_title or "岗位信息", text)
            )
            self._pending_section_title = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize one rendered Baidu job DOM for JobScope."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--company", default="百度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = BaiduJobDomExtractor()
    extractor.feed(args.input.read_text(encoding="utf-8-sig"))
    extractor.close()
    if extractor.title is None or len(extractor.sections) < 2:
        raise ValueError(
            "rendered page does not contain a complete Baidu job detail"
        )

    body = [
        f"<h1>{escape(extractor.title)}</h1>",
        f"<p>企业：{escape(args.company)}</p>",
        f"<p>官方来源：{escape(args.source_url)}</p>",
        f"<p>抓取时间：{escape(args.captured_at)}</p>",
    ]
    body.extend(
        f"<p>{escape(label)}：{escape(item)}</p>"
        for label, item in zip(
            extractor._METADATA_LABELS,
            extractor.metadata,
            strict=False,
        )
    )
    for title, content in extractor.sections:
        body.append(f"<h2>{escape(title)}</h2>")
        for line in content.splitlines():
            body.append(f"<p>{escape(line)}</p>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "<!doctype html><html lang=\"zh-CN\"><head>"
        "<meta charset=\"utf-8\"><title>"
        f"{escape(extractor.title)}</title></head><body>"
        + "\n".join(body)
        + "</body></html>\n",
        encoding="utf-8",
    )
    print(
        f"title={extractor.title} metadata={len(extractor.metadata)} "
        f"sections={len(extractor.sections)} output={args.output}"
    )


if __name__ == "__main__":
    main()
