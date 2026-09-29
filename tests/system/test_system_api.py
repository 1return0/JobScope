import unittest
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.domain.sources.source_trust import (
    VerifiedCompanyDomain,
    VerifiedDomainRegistry,
)
from app.main import app


class SystemApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_returns_service_identity(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "status": "UP",
                "service": "JobScope",
                "version": "0.1.0",
                "environment": "development",
            },
            response.json(),
        )

    def test_meta_distinguishes_implemented_and_planned_capabilities(self) -> None:
        response = self.client.get("/v1/meta")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertCountEqual(
            ["html", "pdf", "docx", "xlsx", "pptx"],
            payload["supported_formats"],
        )
        self.assertEqual("productization", payload["stage"])
        self.assertIn("health-check", payload["implemented_capabilities"])
        self.assertIn(
            "source-schema-validation",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "conservative-source-assessment",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "verified-domain-matching",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "verified-domain-csv-loading",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "postgresql-domain-repository",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "alembic-schema-migrations",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "idempotent-domain-import",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "html-document-parsing",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "docx-document-parsing",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "pdf-text-layer-parsing",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "xlsx-row-parsing",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "pptx-slide-parsing",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "default-multi-format-parser-composition",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "bounded-temporary-document-upload-api",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "stable-evidence-id-contract",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "structure-preserving-character-chunking",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "postgresql-evidence-chunk-repository",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "versioned-document-snapshot-repository",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "atomic-document-corpus-persistence",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "current-version-corpus-read-model",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "in-memory-bm25-lexical-baseline",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "retrieval-evaluation-metrics-contract",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "document-parsing-failure-classification",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "verified-corpus-manifest-loading",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "immutable-corpus-manifest-freezing",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "manifest-driven-corpus-ingestion",
            payload["implemented_capabilities"],
        )
        self.assertIn("scanned-pdf-ocr", payload["implemented_capabilities"])
        self.assertIn("hybrid-retrieval", payload["implemented_capabilities"])
        self.assertIn(
            "evidence-grounded-answers",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "job-agent-planning",
            payload["implemented_capabilities"],
        )
        self.assertIn(
            "frontend-operational-workbench",
            payload["implemented_capabilities"],
        )
        self.assertTrue(payload["read_only"])

    def test_cors_allows_configured_vue_development_origin(self) -> None:
        response = self.client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            "http://localhost:5173",
            response.headers["access-control-allow-origin"],
        )

    def test_source_candidate_validation_accepts_valid_payload(self) -> None:
        response = self.client.post(
            "/v1/source-candidates/validate",
            json={
                "company": "Example Technology",
                "job_title": "AI Agent Intern",
                "source_url": "https://careers.example.com/jobs/agent-intern",
                "source_kind": "official_company",
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "schema_valid": True,
                "normalized_url": (
                    "https://careers.example.com/jobs/agent-intern"
                ),
            },
            response.json(),
        )

    def test_source_candidate_validation_rejects_invalid_payload(self) -> None:
        response = self.client.post(
            "/v1/source-candidates/validate",
            json={
                "company": "",
                "job_title": "AI Agent Intern",
                "source_url": "not-a-url",
                "source_kind": "social_media",
            },
        )

        self.assertEqual(422, response.status_code)
        error_locations = {
            tuple(error["loc"]) for error in response.json()["detail"]
        }
        self.assertTrue(
            {
                ("body", "company"),
                ("body", "source_url"),
                ("body", "source_kind"),
            }.issubset(error_locations)
        )

    def test_source_assessment_does_not_confuse_claim_with_trust(self) -> None:
        response = self.client.post(
            "/v1/source-candidates/assess",
            json={
                "company": "Example Technology",
                "job_title": "AI Agent Intern",
                "source_url": "https://careers.example.com/jobs/agent-intern",
                "source_kind": "official_company",
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "source_kind": "official_company",
                "domain_verified": False,
                "transport_secure": True,
                "content_freshness_verified": False,
                "trust_verified": False,
                "manual_review_required": True,
                "review_reasons": [
                    "company-domain-ownership-not-verified",
                    "content-freshness-not-verified",
                ],
                "matched_official_domain": None,
                "domain_evidence_reference": None,
                "domain_verified_at": None,
            },
            response.json(),
        )

    def test_source_assessment_assigns_reasons_by_source_kind(self) -> None:
        cases = [
            (
                "official_company",
                "https://careers.example.com/jobs/1",
                [
                    "company-domain-ownership-not-verified",
                    "content-freshness-not-verified",
                ],
            ),
            (
                "official_university",
                "https://career.university.example/jobs/1",
                [
                    "secondary-source-requires-cross-check",
                    "content-freshness-not-verified",
                ],
            ),
            (
                "job_board",
                "http://jobs.example.com/jobs/1",
                [
                    "aggregated-source-requires-cross-check",
                    "non-https-url",
                    "content-freshness-not-verified",
                ],
            ),
        ]

        for source_kind, source_url, expected_reasons in cases:
            with self.subTest(source_kind=source_kind):
                response = self.client.post(
                    "/v1/source-candidates/assess",
                    json={
                        "company": "Example Technology",
                        "job_title": "AI Agent Intern",
                        "source_url": source_url,
                        "source_kind": source_kind,
                    },
                )

                self.assertEqual(200, response.status_code)
                payload = response.json()
                self.assertEqual(source_kind, payload["source_kind"])
                self.assertEqual(
                    expected_reasons,
                    payload["review_reasons"],
                )
                self.assertFalse(payload["trust_verified"])
                self.assertTrue(payload["manual_review_required"])
                self.assertIsNone(payload["matched_official_domain"])
                self.assertIsNone(
                    payload["domain_evidence_reference"]
                )
                self.assertIsNone(payload["domain_verified_at"])

    def test_source_assessment_rejects_unknown_source_kind(self) -> None:
        response = self.client.post(
            "/v1/source-candidates/assess",
            json={
                "company": "Example Technology",
                "job_title": "AI Agent Intern",
                "source_url": "https://social.example.com/jobs/1",
                "source_kind": "social_media",
            },
        )

        self.assertEqual(422, response.status_code)
        error_locations = {
            tuple(error["loc"]) for error in response.json()["detail"]
        }
        self.assertIn(("body", "source_kind"), error_locations)

    def test_verified_domain_does_not_overclaim_content_trust(self) -> None:
        registry = VerifiedDomainRegistry(
            [
                VerifiedCompanyDomain(
                    company="Example Technology",
                    domain="example.com",
                    evidence_reference="manual-test-evidence",
                    verified_at=date(2026, 7, 25),
                    verified_by="test-reviewer",
                    status="active",
                )
            ]
        )

        with patch("app.main.verified_domain_registry", registry):
            response = self.client.post(
                "/v1/source-candidates/assess",
                json={
                    "company": "Example Technology",
                    "job_title": "AI Agent Intern",
                    "source_url": (
                        "https://careers.example.com/jobs/agent-intern"
                    ),
                    "source_kind": "official_company",
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "source_kind": "official_company",
                "domain_verified": True,
                "transport_secure": True,
                "content_freshness_verified": False,
                "trust_verified": False,
                "manual_review_required": True,
                "review_reasons": [
                    "content-freshness-not-verified"
                ],
                "matched_official_domain": "example.com",
                "domain_evidence_reference": "manual-test-evidence",
                "domain_verified_at": "2026-07-25",
            },
            response.json(),
        )

if __name__ == "__main__":
    unittest.main()
