import json
import tempfile
import unittest
from pathlib import Path

from app.infrastructure.retrieval.retrieval_evaluation_json import (
    RetrievalEvaluationJsonError,
    load_retrieval_evaluation_dataset,
)


class RetrievalEvaluationJsonTest(unittest.TestCase):
    def test_loads_versioned_dataset(self) -> None:
        path = self._write_dataset()

        dataset = load_retrieval_evaluation_dataset(path)

        self.assertEqual("dataset-v1", dataset.dataset_id)
        self.assertEqual("reviewed", dataset.annotation_status)
        self.assertEqual(1, len(dataset.cases))
        self.assertEqual(3, dataset.cases[0].judgments[0].relevance_grade)

    def test_rejects_non_integer_relevance_grade(self) -> None:
        path = self._write_dataset(relevance_grade="3")

        with self.assertRaisesRegex(
            RetrievalEvaluationJsonError,
            "integer",
        ):
            load_retrieval_evaluation_dataset(path)

    def _write_dataset(self, relevance_grade=3) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "evaluation.json"
        path.write_text(
            json.dumps(
                {
                    "dataset_id": "dataset-v1",
                    "annotation_status": "reviewed",
                    "annotated_by": "student",
                    "source_reference": "official-source",
                    "document_sha256": "a" * 64,
                    "chunker_version": "structure-v1",
                    "tokenizer_version": "mixed-cjk-bigram-v1",
                    "cases": [
                        {
                            "case_id": "case-1",
                            "query": "resume application",
                            "judgments": [
                                {
                                    "evidence_id": "ev_" + "1" * 64,
                                    "relevance_grade": relevance_grade,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
