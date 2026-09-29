import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.infrastructure.reports.ocr_evaluation_dataset_json import (
    OcrEvaluationDatasetFileError,
    load_ocr_evaluation_dataset,
    verify_ocr_evaluation_dataset_files,
)


class OcrEvaluationDatasetJsonTest(unittest.TestCase):
    def test_loads_and_verifies_referenced_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "scan.pdf"
            text_path = root / "scan.expected.txt"
            pdf_path.write_bytes(b"pdf")
            text_path.write_text("AI Agent", encoding="utf-8")
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    _payload(
                        _sha256(pdf_path),
                        _sha256(text_path),
                    )
                ),
                encoding="utf-8",
            )

            dataset = load_ocr_evaluation_dataset(dataset_path)
            verified = verify_ocr_evaluation_dataset_files(
                dataset,
                artifact_root=root,
            )

        self.assertEqual("ocr-v1", dataset.dataset_id)
        self.assertEqual(pdf_path, verified[0].artifact_path)

    def test_rejects_a_changed_ground_truth_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "scan.pdf"
            text_path = root / "scan.expected.txt"
            pdf_path.write_bytes(b"pdf")
            text_path.write_text("AI Agent", encoding="utf-8")
            payload = _payload(
                _sha256(pdf_path),
                _sha256(text_path),
            )
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_ocr_evaluation_dataset(dataset_path)
            text_path.write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(
                OcrEvaluationDatasetFileError,
                "hash mismatch",
            ):
                verify_ocr_evaluation_dataset_files(
                    dataset,
                    artifact_root=root,
                )


def _payload(pdf_hash: str, text_hash: str) -> dict[str, object]:
    return {
        "dataset_id": "ocr-v1",
        "dataset_version": 1,
        "review_status": "reviewed",
        "cases": [
            {
                "case_id": "scan",
                "artifact_reference": "scan.pdf",
                "artifact_sha256": pdf_hash,
                "expected_text_reference": "scan.expected.txt",
                "expected_text_sha256": text_hash,
                "tags": ["synthetic"],
                "fields": [
                    {
                        "field_name": "job_title",
                        "expected_value": "AI Agent",
                        "critical": True,
                    }
                ],
            }
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
