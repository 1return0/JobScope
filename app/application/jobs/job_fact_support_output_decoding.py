from __future__ import annotations

import json

from app.application.jobs.job_record_semantic_support import (
    JobFactSupportDecision,
)


class JobFactSupportOutputDecodingError(ValueError):
    pass


class JobFactSupportOutputDecoder:
    def decode(self, raw_output: str) -> JobFactSupportDecision:
        try:
            payload = json.loads(raw_output)
            if not isinstance(payload, dict):
                raise TypeError("job fact support output must be an object")
            if set(payload) != {"label", "rationale"}:
                raise TypeError(
                    "job fact support output fields do not match contract"
                )
            label = payload["label"]
            rationale = payload["rationale"]
            if label not in (
                "supported",
                "unsupported",
                "uncertain",
            ):
                raise TypeError("job fact support label is invalid")
            if not isinstance(rationale, str):
                raise TypeError(
                    "job fact support rationale must be a string"
                )
            return JobFactSupportDecision(label, rationale)
        except (KeyError, TypeError, ValueError) as error:
            raise JobFactSupportOutputDecodingError(
                "model returned an invalid job fact support decision"
            ) from error
