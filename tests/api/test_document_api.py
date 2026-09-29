import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.documents.document_api import build_document_router
from app.bootstrap import (
    build_document_parsing_service,
    build_document_processing_service,
)


class DocumentApiTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(
            build_document_router(
                build_document_parsing_service(),
                build_document_processing_service(),
                max_upload_bytes=128,
            )
        )
        self.client = TestClient(app)

    def test_parses_upload_and_returns_evidence_location(self) -> None:
        response = self.client.post(
            "/v1/documents/parse",
            data={
                "source_reference": (
                    "https://careers.example/notice.html"
                )
            },
            files={
                "file": (
                    "../../notice.html",
                    (
                        b"<h1>Campus Recruitment</h1>"
                        b"<p>Apply by Friday.</p>"
                    ),
                    "text/html",
                )
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("succeeded", payload["status"])
        self.assertEqual("notice.html", payload["filename"])
        self.assertEqual("html", payload["document_format"])
        self.assertEqual(
            ["Campus Recruitment", "Apply by Friday."],
            [
                fragment["text"]
                for fragment in payload["fragments"]
            ],
        )
        self.assertEqual(
            ["Campus Recruitment"],
            payload["fragments"][1]["location"]["heading_path"],
        )
        self.assertIsNone(payload["failure"])

    def test_rejects_unsupported_extension_before_parsing(self) -> None:
        response = self.client.post(
            "/v1/documents/parse",
            data={"source_reference": "source-text"},
            files={
                "file": (
                    "notice.txt",
                    b"recruitment",
                    "text/plain",
                )
            },
        )

        self.assertEqual(415, response.status_code)
        self.assertEqual(
            "unsupported document extension",
            response.json()["detail"],
        )

    def test_rejects_upload_larger_than_configured_limit(self) -> None:
        response = self.client.post(
            "/v1/documents/parse",
            data={"source_reference": "source-large"},
            files={
                "file": (
                    "notice.html",
                    b"<p>" + b"x" * 200 + b"</p>",
                    "text/html",
                )
            },
        )

        self.assertEqual(413, response.status_code)

    def test_rejects_blank_source_reference(self) -> None:
        response = self.client.post(
            "/v1/documents/parse",
            data={"source_reference": "   "},
            files={
                "file": (
                    "notice.html",
                    b"<p>Recruitment</p>",
                    "text/html",
                )
            },
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual(
            "source_reference must not be blank",
            response.json()["detail"],
        )

    def test_returns_known_parsing_failure_as_stable_response(
        self,
    ) -> None:
        response = self.client.post(
            "/v1/documents/parse",
            data={"source_reference": "source-empty"},
            files={
                "file": (
                    "empty.html",
                    b"<html><body><div></div></body></html>",
                    "text/html",
                )
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("failed", payload["status"])
        self.assertEqual([], payload["fragments"])
        self.assertEqual(
            "no-extractable-content",
            payload["failure"]["code"],
        )
        self.assertEqual(
            "use-fallback-extractor-or-manual-review",
            payload["failure"]["operator_action"],
        )

    def test_processes_upload_into_traceable_evidence_chunks(
        self,
    ) -> None:
        response = self.client.post(
            "/v1/documents/process",
            data={
                "source_reference": (
                    "https://careers.example/notice.html"
                )
            },
            files={
                "file": (
                    "notice.html",
                    (
                        b"<h1>Campus Recruitment</h1>"
                        b"<p>Apply by Friday.</p>"
                    ),
                    "text/html",
                )
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("succeeded", payload["status"])
        self.assertEqual(2, payload["fragment_count"])
        self.assertEqual(2, payload["chunk_count"])
        self.assertEqual(
            ["Campus Recruitment"],
            payload["chunks"][1]["location"]["heading_path"],
        )
        self.assertTrue(
            payload["chunks"][1]["evidence_id"].startswith("ev_")
        )
        self.assertIn(
            "max_chars=800",
            payload["chunks"][1]["chunker_version"],
        )

    def test_processing_same_upload_returns_stable_evidence_ids(
        self,
    ) -> None:
        request = {
            "data": {"source_reference": "source-stable"},
            "files": {
                "file": (
                    "notice.html",
                    b"<p>AI Agent Intern</p>",
                    "text/html",
                )
            },
        }

        first = self.client.post(
            "/v1/documents/process",
            **request,
        )
        second = self.client.post(
            "/v1/documents/process",
            **request,
        )

        self.assertEqual(
            first.json()["chunks"][0]["evidence_id"],
            second.json()["chunks"][0]["evidence_id"],
        )

    def test_processing_failure_does_not_create_chunks(self) -> None:
        response = self.client.post(
            "/v1/documents/process",
            data={"source_reference": "source-empty"},
            files={
                "file": (
                    "empty.html",
                    b"<html><body></body></html>",
                    "text/html",
                )
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("failed", payload["status"])
        self.assertEqual(0, payload["fragment_count"])
        self.assertEqual(0, payload["chunk_count"])
        self.assertEqual([], payload["chunks"])
        self.assertEqual(
            "no-extractable-content",
            payload["failure"]["code"],
        )


if __name__ == "__main__":
    unittest.main()
