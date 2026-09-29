from __future__ import annotations

from typing import Literal, TypeAlias


RecruitmentType: TypeAlias = Literal[
    "internship",
    "campus",
    "experienced",
]


class UnsupportedRecruitmentTypeError(ValueError):
    pass


_ALIASES: dict[RecruitmentType, tuple[str, ...]] = {
    "internship": (
        "internship",
        "intern",
        "实习",
        "实习生",
        "实习招聘",
        "日常实习",
        "日常实习项目",
    ),
    "campus": (
        "campus",
        "campus recruitment",
        "graduate",
        "校招",
        "校园招聘",
        "应届生招聘",
    ),
    "experienced": (
        "experienced",
        "experienced hire",
        "social recruitment",
        "社招",
        "社会招聘",
    ),
}

_CANONICAL_BY_ALIAS = {
    alias.casefold(): canonical
    for canonical, aliases in _ALIASES.items()
    for alias in aliases
}


def normalize_recruitment_type(value: str) -> RecruitmentType:
    normalized = value.strip().casefold()
    if not normalized:
        raise UnsupportedRecruitmentTypeError(
            "recruitment type must not be blank"
        )
    try:
        return _CANONICAL_BY_ALIAS[normalized]
    except KeyError as error:
        raise UnsupportedRecruitmentTypeError(
            f"unsupported recruitment type: {value.strip()}"
        ) from error


def recruitment_type_storage_aliases(
    value: RecruitmentType,
) -> tuple[str, ...]:
    return _ALIASES[value]
