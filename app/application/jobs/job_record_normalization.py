from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.domain.jobs.structured_job_record import EvidenceBackedJobFact


JOB_DATE_NORMALIZER_VERSION = "job-date-v1"

DeadlineNormalizationStatus = Literal[
    "normalized",
    "not_stated",
    "manual_review_required",
]


@dataclass(frozen=True, slots=True)
class NormalizedApplicationDeadline:
    raw_fact: EvidenceBackedJobFact
    normalized_date: date | None
    status: DeadlineNormalizationStatus
    reason_code: str | None
    normalizer_version: str = JOB_DATE_NORMALIZER_VERSION

    def __post_init__(self) -> None:
        if self.status == "normalized":
            if self.normalized_date is None or self.reason_code is not None:
                raise ValueError(
                    "normalized deadline requires a date and no reason code"
                )
            return
        if self.normalized_date is not None:
            raise ValueError(
                "non-normalized deadline must not contain normalized_date"
            )
        if self.status == "not_stated":
            if self.raw_fact.is_stated or self.reason_code is not None:
                raise ValueError(
                    "not-stated deadline must contain an unknown raw fact"
                )
            return
        if not self.raw_fact.is_stated or not (self.reason_code or "").strip():
            raise ValueError(
                "manual-review deadline requires raw value and reason code"
            )


class ApplicationDeadlineNormalizer:
    _PATTERNS = (
        re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$"),
        re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
        re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$"),
    )

    def normalize(
        self,
        fact: EvidenceBackedJobFact,
    ) -> NormalizedApplicationDeadline:
        if not fact.is_stated:
            return NormalizedApplicationDeadline(
                raw_fact=fact,
                normalized_date=None,
                status="not_stated",
                reason_code=None,
            )

        raw_value = fact.value
        if raw_value is None:
            raise AssertionError("stated deadline must contain value")
        for pattern in self._PATTERNS:
            match = pattern.fullmatch(raw_value)
            if match is None:
                continue
            try:
                normalized_date = date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            except ValueError:
                return NormalizedApplicationDeadline(
                    raw_fact=fact,
                    normalized_date=None,
                    status="manual_review_required",
                    reason_code="invalid-calendar-date",
                )
            return NormalizedApplicationDeadline(
                raw_fact=fact,
                normalized_date=normalized_date,
                status="normalized",
                reason_code=None,
            )

        return NormalizedApplicationDeadline(
            raw_fact=fact,
            normalized_date=None,
            status="manual_review_required",
            reason_code="unsupported-date-expression",
        )
