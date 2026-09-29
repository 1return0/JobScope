import csv
import hashlib
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.documents.corpus_manifest import (
    CorpusManifest,
    CorpusManifestEntry,
)
from app.infrastructure.documents.corpus_manifest_csv import (
    MANIFEST_COLUMNS,
    CorpusManifestCsvError,
    load_verified_corpus_manifest,
)


class CorpusManifestTest(unittest.TestCase):
    def _entry(
        self,
        *,
        source_id: str = "example-2027-backend",
    ) -> CorpusManifestEntry:
        return CorpusManifestEntry(
            source_id=source_id,
            company="Example Technology",
            job_title="Backend Engineer",
            job_id="backend-01",
            recruitment_type="campus",
            published_date=None,
            location="Shanghai",
            source_kind="official_company",
            source_url="https://careers.example/jobs/backend",
            final_url="https://careers.example/files/backend.pdf",
            captured_at=datetime(2026, 7, 30, 12, tzinfo=UTC),
            artifact_path=Path("example/backend.pdf"),
            content_sha256="a" * 64,
            status="active",
        )

    def test_manifest_identity_is_stable_across_row_order(self) -> None:
        first = self._entry(source_id="source-a")
        second = self._entry(source_id="source-b")

        self.assertEqual(
            CorpusManifest((first, second)).identity,
            CorpusManifest((second, first)).identity,
        )

    def test_manifest_rejects_duplicate_source_ids(self) -> None:
        first = self._entry()
        duplicate = self._entry()

        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            CorpusManifest((first, duplicate))

    def test_entry_requires_timezone_aware_capture_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            CorpusManifestEntry(
                source_id="example-2027-backend",
                company="Example Technology",
                job_title="Backend Engineer",
                job_id=None,
                recruitment_type="campus",
                published_date=None,
                location="Shanghai",
                source_kind="official_company",
                source_url="https://careers.example/jobs/backend",
                final_url="https://careers.example/files/backend.pdf",
                captured_at=datetime(2026, 7, 30, 12),
                artifact_path=Path("example/backend.pdf"),
                content_sha256="a" * 64,
                status="active",
            )

    def test_entry_rejects_artifact_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifact root"):
            CorpusManifestEntry(
                source_id="example-2027-backend",
                company="Example Technology",
                job_title="Backend Engineer",
                job_id=None,
                recruitment_type="campus",
                published_date=None,
                location="Shanghai",
                source_kind="official_company",
                source_url="https://careers.example/jobs/backend",
                final_url="https://careers.example/files/backend.pdf",
                captured_at=datetime(2026, 7, 30, 12, tzinfo=UTC),
                artifact_path=Path("../outside.pdf"),
                content_sha256="a" * 64,
                status="active",
            )


class CorpusManifestCsvTest(unittest.TestCase):
    def _write_manifest(
        self,
        path: Path,
        *,
        rows: list[dict[str, str]],
        columns: tuple[str, ...] = MANIFEST_COLUMNS,
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _row(
        *,
        content_sha256: str,
        artifact_path: str = "example/backend.pdf",
        source_id: str = "example-2027-backend",
    ) -> dict[str, str]:
        return {
            "source_id": source_id,
            "company": "Example Technology",
            "job_title": "Backend Engineer",
            "job_id": "backend-01",
            "recruitment_type": "campus",
            "published_date": "",
            "location": "Shanghai",
            "source_kind": "official_company",
            "source_url": "https://careers.example/jobs/backend",
            "final_url": (
                "https://careers.example/files/backend.pdf"
            ),
            "captured_at": "2026-07-30T12:00:00+08:00",
            "artifact_path": artifact_path,
            "content_sha256": content_sha256,
            "status": "active",
            "notes": "",
        }

    def test_loads_manifest_only_after_rechecking_artifact_hash(
        self,
    ) -> None:
        content = b"official recruitment document"
        content_sha256 = hashlib.sha256(content).hexdigest()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            artifact_path = artifact_root / "example" / "backend.pdf"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(content)
            manifest_path = root / "manifest.csv"
            self._write_manifest(
                manifest_path,
                rows=[self._row(content_sha256=content_sha256)],
            )

            verified = load_verified_corpus_manifest(
                manifest_path,
                artifact_root=artifact_root,
            )

        self.assertEqual(1, len(verified.documents))
        self.assertEqual(1, verified.manifest.company_count)
        self.assertEqual(
            content_sha256,
            verified.documents[0].artifact.content_sha256,
        )
        self.assertEqual(
            "https://careers.example/files/backend.pdf",
            verified.documents[0].artifact.source_reference,
        )

    def test_rejects_artifact_changed_after_manifest_recorded(
        self,
    ) -> None:
        original_sha256 = hashlib.sha256(b"original").hexdigest()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            artifact_path = artifact_root / "example" / "backend.pdf"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"changed")
            manifest_path = root / "manifest.csv"
            self._write_manifest(
                manifest_path,
                rows=[self._row(content_sha256=original_sha256)],
            )

            with self.assertRaisesRegex(
                CorpusManifestCsvError,
                "hash mismatch",
            ):
                load_verified_corpus_manifest(
                    manifest_path,
                    artifact_root=artifact_root,
                )

    def test_rejects_manifest_schema_drift(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            artifact_root.mkdir()
            manifest_path = root / "manifest.csv"
            self._write_manifest(
                manifest_path,
                rows=[],
                columns=MANIFEST_COLUMNS[:-1],
            )

            with self.assertRaisesRegex(
                CorpusManifestCsvError,
                "columns",
            ):
                load_verified_corpus_manifest(
                    manifest_path,
                    artifact_root=artifact_root,
                )


if __name__ == "__main__":
    unittest.main()
