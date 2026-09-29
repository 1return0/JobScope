from __future__ import annotations

import json

from app.application.answering.claim_support_evaluation import (
    ClaimSupportJudgment,
)


class ClaimSupportOutputDecodingError(ValueError):
    failure_code = "claim-support-output-invalid"


class ClaimSupportOutputDecoder:
    def decode(self, raw_output: str) -> ClaimSupportJudgment:
        try:
            payload = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError) as error:
            raise ClaimSupportOutputDecodingError(
                "claim-support output is not valid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise ClaimSupportOutputDecodingError(
                "claim-support output must be a JSON object"
            )
        try:
            label = payload["label"]
            rationale = payload["rationale"]
            if not isinstance(label, str) or not isinstance(
                rationale,
                str,
            ):
                raise TypeError("label and rationale must be strings")
            return ClaimSupportJudgment(
                label=label,  # type: ignore[arg-type]
                rationale=rationale,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ClaimSupportOutputDecodingError(
                "claim-support output does not match the contract"
            ) from error
