import csv
import hashlib
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from app.infrastructure.documents.corpus_manifest_csv import (
    MANIFEST_COLUMNS,
    CorpusManifestCsvError,
    load_verified_corpus_manifest,
    register_corpus_artifact,
)


class CorpusArtifactRegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.artifact_root = self.root / "artifacts"
        self.artifact_root.mkdir()
        self.manifest_path = self.root / "manifest.csv"
        with self.manifest_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as manifest_file:
            csv.DictWriter(
                manifest_file,
                fieldnames=MANIFEST_COLUMNS,
            ).writeheader()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_computes_technical_fields_and_registers_entry(self) -> None:
        relative_path = Path("example/backend.html")
        artifact_path = self.artifact_root / relative_path
        artifact_path.parent.mkdir()
        content = b"<h1>Backend Engineer</h1><p>Java required.</p>"
        artifact_path.write_bytes(content)

        report = self._register(relative_path)

        expected_hash = hashlib.sha256(content).hexdigest()
        self.assertEqual(expected_hash, report.entry.content_sha256)
        self.assertEqual("html", report.document_format)
        self.assertEqual(len(content), report.byte_size)
        self.assertEqual(1, report.document_count)
        self.assertEqual("created", report.operation)
        self.assertFalse(report.content_changed)
        verified = load_verified_corpus_manifest(
            self.manifest_path,
            artifact_root=self.artifact_root,
        )
        self.assertEqual(expected_hash, verified.documents[0].artifact.content_sha256)

    def test_duplicate_source_id_does_not_modify_manifest(self) -> None:
        first_path = Path("example/first.html")
        second_path = Path("example/second.html")
        (self.artifact_root / "example").mkdir()
        (self.artifact_root / first_path).write_text(
            "first",
            encoding="utf-8",
        )
        (self.artifact_root / second_path).write_text(
            "second",
            encoding="utf-8",
        )
        self._register(first_path)
        original_csv = self.manifest_path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            self._register(second_path)

        self.assertEqual(
            original_csv,
            self.manifest_path.read_text(encoding="utf-8"),
        )

    def test_rejects_artifact_outside_configured_root(self) -> None:
        outside_path = self.root / "outside.html"
        outside_path.write_text("outside", encoding="utf-8")

        with self.assertRaisesRegex(
            CorpusManifestCsvError,
            "escapes configured root",
        ):
            self._register(Path("../outside.html"))

    def test_replaces_source_with_explicitly_verified_new_version(
        self,
    ) -> None:
        first_path = Path("example/backend-v1.html")
        second_path = Path("example/backend-v2.html")
        (self.artifact_root / "example").mkdir()
        first_content = b"<p>Java required.</p>"
        second_content = b"<p>Java and PostgreSQL required.</p>"
        (self.artifact_root / first_path).write_bytes(first_content)
        (self.artifact_root / second_path).write_bytes(second_content)
        first_report = self._register(first_path)

        updated_report = self._register(
            second_path,
            captured_at=datetime(2026, 8, 2, tzinfo=UTC),
            replace_existing=True,
        )
        verified = load_verified_corpus_manifest(
            self.manifest_path,
            artifact_root=self.artifact_root,
        )

        self.assertEqual("updated", updated_report.operation)
        self.assertTrue(updated_report.content_changed)
        self.assertEqual(
            first_report.entry.content_sha256,
            updated_report.previous_content_sha256,
        )
        self.assertNotEqual(
            first_report.manifest_identity,
            updated_report.manifest_identity,
        )
        self.assertEqual(1, len(verified.documents))
        self.assertEqual(
            second_path,
            verified.documents[0].entry.artifact_path,
        )
        self.assertTrue((self.artifact_root / first_path).is_file())

    def test_update_failure_does_not_modify_manifest(self) -> None:
        first_path = Path("example/backend-v1.html")
        second_path = Path("example/backend-v2.html")
        (self.artifact_root / "example").mkdir()
        (self.artifact_root / first_path).write_text(
            "first",
            encoding="utf-8",
        )
        (self.artifact_root / second_path).write_text(
            "second",
            encoding="utf-8",
        )
        self._register(first_path)
        original_csv = self.manifest_path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(
            CorpusManifestCsvError,
            "later captured_at",
        ):
            self._register(
                second_path,
                captured_at=datetime(2026, 8, 1, tzinfo=UTC),
                replace_existing=True,
            )

        self.assertEqual(
            original_csv,
            self.manifest_path.read_text(encoding="utf-8"),
        )

    def _register(
        self,
        artifact_path: Path,
        *,
        captured_at: datetime = datetime(2026, 8, 1, tzinfo=UTC),
        replace_existing: bool = False,
    ):
        return register_corpus_artifact(
            self.manifest_path,
            artifact_root=self.artifact_root,
            artifact_path=artifact_path,
            source_id="example-2027-backend",
            company="Example Technology",
            job_title="Backend Engineer",
            job_id=None,
            recruitment_type="campus",
            published_date=None,
            location="Shanghai",
            source_kind="official_company",
            source_url="https://careers.example/jobs/backend",
            final_url="https://careers.example/jobs/backend.html",
            captured_at=captured_at,
            status="active",
            replace_existing=replace_existing,
        )


if __name__ == "__main__":
    unittest.main()
