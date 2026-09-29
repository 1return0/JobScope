import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.evaluation.evaluation_api import build_evaluation_router


class EvaluationApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        evaluation_dir = self.data_root / "evaluation"
        experiment_dir = self.data_root / "experiments" / "planner"
        evaluation_dir.mkdir(parents=True)
        experiment_dir.mkdir(parents=True)
        (evaluation_dir / "retrieval-v1.json").write_text(
            json.dumps(
                {
                    "dataset_id": "retrieval-v1",
                    "dataset_version": 1,
                    "review_status": "provisional",
                    "cases": [{"case_id": "case-1"}],
                }
            ),
            encoding="utf-8",
        )
        (experiment_dir / "run-01.json").write_text(
            json.dumps(
                {
                    "dataset_id": "planner-v1",
                    "executed_at": "2026-08-24T00:00:00+00:00",
                    "case_count": 2,
                    "metrics": {"case_accuracy": 0.5},
                }
            ),
            encoding="utf-8",
        )
        app = FastAPI()
        app.include_router(build_evaluation_router(self.data_root))
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_catalog_summarizes_datasets_and_experiments(self) -> None:
        response = self.client.get("/v1/evaluations/catalog")

        self.assertEqual(200, response.status_code)
        artifacts = response.json()["artifacts"]
        self.assertEqual(2, len(artifacts))
        experiment = next(
            item for item in artifacts
            if item["artifact_type"] == "experiment"
        )
        self.assertEqual(0.5, experiment["metrics"]["case_accuracy"])
        dataset = next(
            item for item in artifacts
            if item["artifact_type"] == "dataset"
        )
        self.assertEqual(1, dataset["case_count"])

    def test_artifact_detail_uses_catalog_identity(self) -> None:
        catalog = self.client.get("/v1/evaluations/catalog").json()
        artifact_id = catalog["artifacts"][0]["artifact_id"]

        response = self.client.get(
            f"/v1/evaluations/artifacts/{artifact_id}"
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(artifact_id, response.json()["summary"]["artifact_id"])
        self.assertIn("dataset_id", response.json()["payload"])

    def test_unknown_artifact_returns_404(self) -> None:
        response = self.client.get(
            "/v1/evaluations/artifacts/does-not-exist"
        )

        self.assertEqual(404, response.status_code)


if __name__ == "__main__":
    unittest.main()
