import unittest

from app.application.retrieval.lexical_retrieval import (
    Bm25Config,
    Bm25Retriever,
    tokenize_lexical,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class LexicalTokenizerTest(unittest.TestCase):
    def test_normalizes_english_and_uses_cjk_bigrams(self) -> None:
        tokens = tokenize_lexical(
            "Python RAG，学历要求本科及以上"
        )

        self.assertIn("python", tokens)
        self.assertIn("rag", tokens)
        self.assertIn("学历", tokens)
        self.assertIn("要求", tokens)


class Bm25RetrieverTest(unittest.TestCase):
    def test_rare_and_repeated_terms_rank_matching_chunk_first(
        self,
    ) -> None:
        retriever = Bm25Retriever(
            [
                self._chunk(
                    1,
                    "Python Python RAG Agent internship",
                ),
                self._chunk(2, "Python Java backend internship"),
                self._chunk(3, "Java Spring backend internship"),
            ]
        )

        results = retriever.search("Python RAG", top_k=2)

        self.assertEqual([1, 2], [item.rank for item in results])
        self.assertEqual(
            "ev_" + f"{1:064x}",
            results[0].chunk.evidence_id,
        )
        self.assertEqual(
            ("python", "rag"),
            results[0].matched_terms,
        )
        self.assertGreater(results[0].score, results[1].score)

    def test_chinese_query_matches_cjk_business_terms(self) -> None:
        retriever = Bm25Retriever(
            [
                self._chunk(1, "学历要求本科及以上"),
                self._chunk(2, "工作地点上海"),
            ]
        )

        results = retriever.search("学历要求")

        self.assertEqual(1, len(results))
        self.assertEqual(
            "学历要求本科及以上",
            results[0].chunk.text,
        )
        self.assertIn("学历", results[0].matched_terms)

    def test_no_lexical_overlap_returns_no_results(self) -> None:
        retriever = Bm25Retriever(
            [self._chunk(1, "Java Spring backend")]
        )

        self.assertEqual([], retriever.search("RAG Agent"))

    def test_top_k_and_ties_are_deterministic(self) -> None:
        retriever = Bm25Retriever(
            [
                self._chunk(2, "Python"),
                self._chunk(1, "Python"),
                self._chunk(3, "Python"),
            ]
        )

        results = retriever.search("Python", top_k=2)

        self.assertEqual(
            [
                "ev_" + f"{1:064x}",
                "ev_" + f"{2:064x}",
            ],
            [item.chunk.evidence_id for item in results],
        )

    def test_rejects_invalid_query_top_k_and_config(self) -> None:
        retriever = Bm25Retriever(
            [self._chunk(1, "Python")]
        )

        with self.assertRaisesRegex(ValueError, "blank"):
            retriever.search("   ")
        with self.assertRaisesRegex(ValueError, "positive"):
            retriever.search("Python", top_k=0)
        with self.assertRaisesRegex(ValueError, "between"):
            Bm25Config(b=1.5)

    def test_rejects_duplicate_evidence_ids(self) -> None:
        chunk = self._chunk(1, "Python")

        with self.assertRaisesRegex(ValueError, "duplicate"):
            Bm25Retriever([chunk, chunk])

    @staticmethod
    def _chunk(index: int, text: str) -> EvidenceChunk:
        return EvidenceChunk(
            evidence_id="ev_" + f"{index:064x}",
            document_sha256=f"{index:064x}",
            source_reference=f"source-{index}",
            source_fragment_ordinal=0,
            chunk_ordinal=0,
            chunker_version=(
                "structure-v1;max_chars=800;overlap_chars=100"
            ),
            text=text,
            location=EvidenceLocation(page_number=index),
        )


if __name__ == "__main__":
    unittest.main()
