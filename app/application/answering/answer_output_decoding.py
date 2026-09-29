from __future__ import annotations

import json

from app.application.answering.grounded_answering import (
    EvidenceCitation,
    GroundedAnswerDraft,
    GroundedClaim,
)


class AnswerOutputDecodingError(ValueError):
    failure_code: str


class InvalidAnswerJsonError(AnswerOutputDecodingError):
    failure_code = "invalid-answer-json"


class InvalidAnswerContractError(AnswerOutputDecodingError):
    failure_code = "invalid-answer-contract"


class GroundedAnswerOutputDecoder:
    def decode(self, raw_output: str) -> GroundedAnswerDraft:
        if not raw_output.strip():
            raise InvalidAnswerJsonError("model output must not be blank")
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise InvalidAnswerJsonError(
                "model output is not valid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise InvalidAnswerContractError(
                "answer payload must be a JSON object"
            )

        try:
            status = _require_string(payload, "status")
            claims_payload = _require_list(payload, "claims")
            claims = tuple(
                _decode_claim(claim_payload)
                for claim_payload in claims_payload
            )
            fallback_message = payload.get("fallback_message")
            if fallback_message is not None and not isinstance(
                fallback_message,
                str,
            ):
                raise ValueError("fallback_message must be a string or null")
            return GroundedAnswerDraft(
                status=status,  # type: ignore[arg-type]
                claims=claims,
                fallback_message=fallback_message,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidAnswerContractError(
                "model JSON does not match the grounded answer contract"
            ) from error


def _decode_claim(payload: object) -> GroundedClaim:
    if not isinstance(payload, dict):
        raise TypeError("claim must be an object")
    citations_payload = _require_list(payload, "citations")
    return GroundedClaim(
        text=_require_string(payload, "text"),
        citations=tuple(
            _decode_citation(citation_payload)
            for citation_payload in citations_payload
        ),
    )


def _decode_citation(payload: object) -> EvidenceCitation:
    if not isinstance(payload, dict):
        raise TypeError("citation must be an object")
    return EvidenceCitation(
        evidence_id=_require_string(payload, "evidence_id"),
        quoted_text=_require_string(payload, "quoted_text"),
    )


def _require_string(payload: dict[object, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _require_list(payload: dict[object, object], field: str) -> list[object]:
    value = payload[field]
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value
