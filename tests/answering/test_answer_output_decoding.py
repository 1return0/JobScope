import json
import unittest

from app.application.answering.answer_output_decoding import (
    GroundedAnswerOutputDecoder,
    InvalidAnswerContractError,
    InvalidAnswerJsonError,
)


class GroundedAnswerOutputDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = GroundedAnswerOutputDecoder()

    def test_decodes_answered_model_json(self) -> None:
        raw_output = json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "该岗位专业不限。",
                        "citations": [
                            {
                                "evidence_id": "ev_" + "1" * 64,
                                "quoted_text": "专业不限",
                            }
                        ],
                    }
                ],
                "fallback_message": None,
            },
            ensure_ascii=False,
        )

        draft = self.decoder.decode(raw_output)

        self.assertEqual("answered", draft.status)
        self.assertEqual("该岗位专业不限。", draft.claims[0].text)
        self.assertEqual("专业不限", draft.claims[0].citations[0].quoted_text)

    def test_decodes_insufficient_evidence_model_json(self) -> None:
        raw_output = json.dumps(
            {
                "status": "insufficient_evidence",
                "claims": [],
                "fallback_message": "当前证据不足。",
            },
            ensure_ascii=False,
        )

        draft = self.decoder.decode(raw_output)

        self.assertEqual("insufficient_evidence", draft.status)
        self.assertEqual((), draft.claims)

    def test_classifies_invalid_json_separately(self) -> None:
        with self.assertRaises(InvalidAnswerJsonError) as context:
            self.decoder.decode("不是JSON")

        self.assertEqual("invalid-answer-json", context.exception.failure_code)

    def test_rejects_json_with_wrong_contract(self) -> None:
        raw_output = json.dumps(
            {
                "status": "answered",
                "claims": "该岗位专业不限",
                "fallback_message": None,
            },
            ensure_ascii=False,
        )

        with self.assertRaises(InvalidAnswerContractError) as context:
            self.decoder.decode(raw_output)

        self.assertEqual(
            "invalid-answer-contract",
            context.exception.failure_code,
        )

    def test_decoder_does_not_replace_grounding_validation(self) -> None:
        raw_output = json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "编造的结论。",
                        "citations": [
                            {
                                "evidence_id": "ev_" + "9" * 64,
                                "quoted_text": "编造的原文",
                            }
                        ],
                    }
                ],
                "fallback_message": None,
            },
            ensure_ascii=False,
        )

        draft = self.decoder.decode(raw_output)

        self.assertEqual("ev_" + "9" * 64, draft.claims[0].citations[0].evidence_id)


if __name__ == "__main__":
    unittest.main()
