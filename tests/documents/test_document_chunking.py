import unittest
from pathlib import Path

from app.domain.documents.document_chunking import (
    ChunkingConfig,
    StructureAwareChunker,
    create_evidence_chunk,
)
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    EvidenceLocation,
    ParsedDocument,
    ParsedFragment,
)


class EvidenceChunkContractTest(unittest.TestCase):
    def test_same_evidence_is_stable_across_temporary_paths(self) -> None:
        first_document = self._parsed_document(
            path=Path("temporary-a/notice.pdf")
        )
        second_document = self._parsed_document(
            path=Path("temporary-b/renamed.pdf")
        )
        first = create_evidence_chunk(
            first_document,
            source_fragment=first_document.fragments[0],
            chunk_ordinal=0,
            text="AI Agent Intern",
            chunker_version="structure-v1",
        )
        second = create_evidence_chunk(
            second_document,
            source_fragment=second_document.fragments[0],
            chunk_ordinal=0,
            text="AI Agent Intern",
            chunker_version="structure-v1",
        )

        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(67, len(first.evidence_id))

    def test_different_source_reference_changes_evidence_id(self) -> None:
        first_document = self._parsed_document(
            source_reference="https://company.example/notice.pdf"
        )
        second_document = self._parsed_document(
            source_reference="https://university.example/notice.pdf"
        )

        first = self._chunk(first_document)
        second = self._chunk(second_document)

        self.assertNotEqual(first.evidence_id, second.evidence_id)

    def test_identity_changes_when_evidence_coordinates_change(
        self,
    ) -> None:
        page_three = self._chunk(
            self._parsed_document(
                location=EvidenceLocation(page_number=3)
            ),
        )
        page_four = self._chunk(
            self._parsed_document(
                location=EvidenceLocation(page_number=4)
            ),
        )

        self.assertNotEqual(
            page_three.evidence_id,
            page_four.evidence_id,
        )

    def test_chunker_version_change_produces_new_evidence_id(
        self,
    ) -> None:
        parsed_document = self._parsed_document()

        version_one = self._chunk(
            parsed_document,
            chunker_version="structure-v1",
        )
        version_two = self._chunk(
            parsed_document,
            chunker_version="structure-v2",
        )

        self.assertNotEqual(
            version_one.evidence_id,
            version_two.evidence_id,
        )

    def test_chunk_text_is_normalized_before_identity_is_built(
        self,
    ) -> None:
        parsed_document = self._parsed_document()

        spaced = self._chunk(
            parsed_document,
            text="AI   Agent\nIntern",
        )
        normalized = self._chunk(
            parsed_document,
            text="AI Agent Intern",
        )

        self.assertEqual(
            spaced.evidence_id,
            normalized.evidence_id,
        )
        self.assertEqual("AI Agent Intern", spaced.text)

    def test_rejects_fragment_not_owned_by_parsed_document(self) -> None:
        parsed_document = self._parsed_document()
        external_fragment = ParsedFragment(
            ordinal=0,
            text="AI Agent Intern",
            location=EvidenceLocation(page_number=3),
        )

        with self.assertRaisesRegex(ValueError, "must belong"):
            create_evidence_chunk(
                parsed_document,
                source_fragment=external_fragment,
                chunk_ordinal=0,
                text=external_fragment.text,
                chunker_version="structure-v1",
            )

    @staticmethod
    def _parsed_document(
        *,
        path: Path = Path("temporary/notice.pdf"),
        source_reference: str = (
            "https://company.example/notice.pdf"
        ),
        location: EvidenceLocation = EvidenceLocation(
            page_number=3
        ),
    ) -> ParsedDocument:
        artifact = DocumentArtifact(
            path=path,
            source_reference=source_reference,
            document_format="pdf",
            content_sha256="a" * 64,
            byte_size=1024,
        )
        return ParsedDocument(
            artifact=artifact,
            fragments=(
                ParsedFragment(
                    ordinal=0,
                    text="AI Agent Intern",
                    location=location,
                ),
            ),
        )

    @staticmethod
    def _chunk(
        parsed_document: ParsedDocument,
        *,
        text: str = "AI Agent Intern",
        chunker_version: str = "structure-v1",
    ):
        return create_evidence_chunk(
            parsed_document,
            source_fragment=parsed_document.fragments[0],
            chunk_ordinal=0,
            text=text,
            chunker_version=chunker_version,
        )


class StructureAwareChunkerTest(unittest.TestCase):
    def test_short_fragment_becomes_one_chunk_with_same_location(
        self,
    ) -> None:
        parsed_document = self._parsed_document(
            [
                ParsedFragment(
                    ordinal=0,
                    text="AI Agent Intern",
                    location=EvidenceLocation(page_number=3),
                )
            ]
        )

        chunks = StructureAwareChunker(
            ChunkingConfig(max_chars=30, overlap_chars=5)
        ).chunk(parsed_document)

        self.assertEqual(1, len(chunks))
        self.assertEqual("AI Agent Intern", chunks[0].text)
        self.assertEqual(3, chunks[0].location.page_number)
        self.assertEqual(0, chunks[0].source_fragment_ordinal)
        self.assertEqual(0, chunks[0].chunk_ordinal)

    def test_long_fragment_is_split_without_losing_page_boundary(
        self,
    ) -> None:
        parsed_document = self._parsed_document(
            [
                ParsedFragment(
                    ordinal=0,
                    text="甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉",
                    location=EvidenceLocation(page_number=8),
                )
            ]
        )
        config = ChunkingConfig(max_chars=10, overlap_chars=3)

        chunks = StructureAwareChunker(config).chunk(parsed_document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(len(chunk.text) <= 10 for chunk in chunks)
        )
        self.assertTrue(
            all(chunk.location.page_number == 8 for chunk in chunks)
        )
        self.assertEqual(
            list(range(len(chunks))),
            [chunk.chunk_ordinal for chunk in chunks],
        )
        self.assertEqual(
            chunks[0].text[-3:],
            chunks[1].text[:3],
        )

    def test_sentence_boundary_is_preferred_over_hard_cut(self) -> None:
        parsed_document = self._parsed_document(
            [
                ParsedFragment(
                    ordinal=0,
                    text="第一句介绍校园招聘。第二句说明岗位要求很重要。",
                    location=EvidenceLocation(
                        heading_path=("Campus Recruitment",)
                    ),
                )
            ]
        )

        chunks = StructureAwareChunker(
            ChunkingConfig(max_chars=15, overlap_chars=2)
        ).chunk(parsed_document)

        self.assertTrue(chunks[0].text.endswith("。"))
        self.assertEqual(
            ("Campus Recruitment",),
            chunks[0].location.heading_path,
        )

    def test_chunker_never_merges_different_source_fragments(
        self,
    ) -> None:
        parsed_document = self._parsed_document(
            [
                ParsedFragment(
                    ordinal=0,
                    text="Page one",
                    location=EvidenceLocation(page_number=1),
                ),
                ParsedFragment(
                    ordinal=1,
                    text="Page two",
                    location=EvidenceLocation(page_number=2),
                ),
            ]
        )

        chunks = StructureAwareChunker(
            ChunkingConfig(max_chars=100, overlap_chars=10)
        ).chunk(parsed_document)

        self.assertEqual(2, len(chunks))
        self.assertEqual(
            [1, 2],
            [chunk.location.page_number for chunk in chunks],
        )
        self.assertEqual(
            [0, 1],
            [chunk.source_fragment_ordinal for chunk in chunks],
        )

    def test_configuration_is_part_of_chunker_identity(self) -> None:
        first = ChunkingConfig(max_chars=100, overlap_chars=10)
        second = ChunkingConfig(max_chars=120, overlap_chars=10)

        self.assertNotEqual(first.identity, second.identity)
        self.assertIn("max_chars=100", first.identity)
        self.assertIn("overlap_chars=10", first.identity)

    def test_rejects_overlap_equal_to_maximum_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller"):
            ChunkingConfig(max_chars=100, overlap_chars=100)

    @staticmethod
    def _parsed_document(
        fragments: list[ParsedFragment],
    ) -> ParsedDocument:
        return ParsedDocument(
            artifact=DocumentArtifact(
                path=Path("temporary/notice.pdf"),
                source_reference=(
                    "https://company.example/notice.pdf"
                ),
                document_format="pdf",
                content_sha256="b" * 64,
                byte_size=2048,
            ),
            fragments=tuple(fragments),
        )


if __name__ == "__main__":
    unittest.main()
