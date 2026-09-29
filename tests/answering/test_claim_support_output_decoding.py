import unittest

from app.application.answering.claim_support_output_decoding import (
    ClaimSupportOutputDecoder,
    ClaimSupportOutputDecodingError,
)


class ClaimSupportOutputDecoderTest(unittest.TestCase):
    def test_decodes_supported_judgment(self) -> None:
        judgment = ClaimSupportOutputDecoder().decode(
            '{"label":"supported","rationale":"directly stated"}'
        )

        self.assertEqual("supported", judgment.label)
        self.assertEqual("directly stated", judgment.rationale)

    def test_rejects_unknown_label(self) -> None:
        with self.assertRaises(ClaimSupportOutputDecodingError):
            ClaimSupportOutputDecoder().decode(
                '{"label":"maybe","rationale":"uncertain"}'
            )

    def test_rejects_non_json_output(self) -> None:
        with self.assertRaises(ClaimSupportOutputDecodingError):
            ClaimSupportOutputDecoder().decode("supported")


if __name__ == "__main__":
    unittest.main()
