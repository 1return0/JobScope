from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TypeVar


_Token = TypeVar("_Token")


@dataclass(frozen=True, slots=True)
class OcrTextEvaluation:
    normalized_expected: str
    normalized_actual: str
    exact_match: bool
    character_edit_count: int
    expected_character_count: int
    character_error_rate: float
    word_edit_count: int
    expected_word_count: int
    word_error_rate: float


def evaluate_ocr_text(
    expected: str,
    actual: str,
) -> OcrTextEvaluation:
    normalized_expected = _normalize(expected)
    normalized_actual = _normalize(actual)
    if not normalized_expected:
        raise ValueError("expected OCR text must not be blank")

    expected_characters = tuple(normalized_expected)
    actual_characters = tuple(normalized_actual)
    expected_words = tuple(normalized_expected.split())
    actual_words = tuple(normalized_actual.split())
    character_edit_count = _edit_distance(
        expected_characters,
        actual_characters,
    )
    word_edit_count = _edit_distance(
        expected_words,
        actual_words,
    )
    return OcrTextEvaluation(
        normalized_expected=normalized_expected,
        normalized_actual=normalized_actual,
        exact_match=normalized_expected == normalized_actual,
        character_edit_count=character_edit_count,
        expected_character_count=len(expected_characters),
        character_error_rate=(
            character_edit_count / len(expected_characters)
        ),
        word_edit_count=word_edit_count,
        expected_word_count=len(expected_words),
        word_error_rate=word_edit_count / len(expected_words),
    )


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _edit_distance(
    expected: tuple[_Token, ...],
    actual: tuple[_Token, ...],
) -> int:
    previous_row = list(range(len(actual) + 1))
    for expected_index, expected_token in enumerate(expected, start=1):
        current_row = [expected_index]
        for actual_index, actual_token in enumerate(actual, start=1):
            deletion = previous_row[actual_index] + 1
            insertion = current_row[actual_index - 1] + 1
            substitution = (
                previous_row[actual_index - 1]
                + (expected_token != actual_token)
            )
            current_row.append(min(deletion, insertion, substitution))
        previous_row = current_row
    return previous_row[-1]
