import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.documents.corpus_manifest import CorpusManifest
from app.infrastructure.documents.corpus_manifest_csv import (
    MANIFEST_COLUMNS,
    VerifiedCorpusManifest,
    load_verified_corpus_manifest,
)
from app.infrastructure.documents.corpus_manifest_freeze import (
    EmptyCorpusManifestError,
    FROZEN_MANIFEST_SCHEMA_VERSION,
    FrozenManifestConflictError,
    freeze_verified_corpus_manifest,
)


class CorpusManifestFreezeTest(unittest.TestCase):
    def _verified_manifest(self, root: Path):
        content = b"official recruitment document"
        content_sha256 = hashlib.sha256(content).hexdigest()
        artifact_root = root / "artifacts"
        artifact_path = artifact_root / "example" / "backend.pdf"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(content)

        manifest_path = root / "manifest.csv"
        row = {
            "source_id": "example-2027-backend",
            "company": "Example Technology",
            "job_title": "Backend Engineer",
            "job_id": "backend-01",
            "recruitment_type": "campus",
            "published_date": "",
            "location": "Shanghai",
            "source_kind": "official_company",
            "source_url": "https://careers.example/jobs/backend",
            "final_url": "https://careers.example/files/backend.pdf",
            "captured_at": "2026-08-01T10:00:00+08:00",
            "artifact_path": "example/backend.pdf",
            "content_sha256": content_sha256,
            "status": "active",
            "notes": "",
        }
        with manifest_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=MANIFEST_COLUMNS,
            )
            writer.writeheader()
            writer.writerow(row)

        return load_verified_corpus_manifest(
            manifest_path,
            artifact_root=artifact_root,
        )

    def test_freezes_verified_manifest_as_identity_named_json(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            verified = self._verified_manifest(root)

            report = freeze_verified_corpus_manifest(
                verified,
                output_directory=root / "frozen",
            )
            content = json.loads(report.path.read_text("utf-8"))

        self.assertTrue(report.created)
        self.assertEqual(
            f"{verified.manifest.identity}.json",
            report.path.name,
        )
        self.assertEqual(
            FROZEN_MANIFEST_SCHEMA_VERSION,
            content["schema_version"],
        )
        self.assertEqual(1, content["document_count"])
        self.assertEqual(
            "pdf",
            content["documents"][0]["artifact"][
                "document_format"
            ],
        )

    def test_repeated_freeze_reuses_identical_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            verified = self._verified_manifest(root)
            output_directory = root / "frozen"

            first = freeze_verified_corpus_manifest(
                verified,
                output_directory=output_directory,
            )
            original_bytes = first.path.read_bytes()
            second = freeze_verified_corpus_manifest(
                verified,
                output_directory=output_directory,
            )
            repeated_bytes = second.path.read_bytes()

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.path, second.path)
        self.assertEqual(original_bytes, repeated_bytes)

    def test_refuses_to_overwrite_conflicting_frozen_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            verified = self._verified_manifest(root)
            output_directory = root / "frozen"
            output_directory.mkdir()
            output_path = (
                output_directory
                / f"{verified.manifest.identity}.json"
            )
            output_path.write_text("corrupt", encoding="utf-8")

            with self.assertRaisesRegex(
                FrozenManifestConflictError,
                "different content",
            ):
                freeze_verified_corpus_manifest(
                    verified,
                    output_directory=output_directory,
                )

    def test_refuses_to_freeze_empty_manifest(self) -> None:
        empty = VerifiedCorpusManifest(
            manifest=CorpusManifest(()),
            documents=(),
        )

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                EmptyCorpusManifestError,
                "empty",
            ):
                freeze_verified_corpus_manifest(
                    empty,
                    output_directory=Path(directory),
                )


if __name__ == "__main__":
    unittest.main()
