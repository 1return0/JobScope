from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic image-only PDF for the local OCR smoke test."
        )
    )
    parser.add_argument("output_path", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    image = Image.new("RGB", (1240, 1754), "white")
    try:
        draw = ImageDraw.Draw(image)
        title_font = _load_font(52)
        body_font = _load_font(38)
        draw.text(
            (100, 120),
            "JOB SCOPE OCR SMOKE TEST",
            fill="black",
            font=title_font,
        )
        lines = (
            "Position: AI Agent Intern",
            "Location: Shanghai",
            "Major: No restriction",
            "Deadline: 2026-09-30",
            "Source: Synthetic student fixture",
        )
        for index, line in enumerate(lines):
            draw.text(
                (120, 300 + index * 110),
                line,
                fill="black",
                font=body_font,
            )

        output = BytesIO()
        image.save(output, format="PDF", resolution=150)
        arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output_path.open("xb") as pdf_file:
            pdf_file.write(output.getvalue())
    finally:
        image.close()

    print(arguments.output_path.resolve())
    return 0


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
