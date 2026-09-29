import unittest

from app.application.retrieval.dense_retrieval import EmbeddingSpec
from app.application.retrieval.hybrid_retrieval_experiment import (
    CurrentCorpusHybridEvaluator,
)
from app.application.retrieval.retrieval_evaluation import (
    GoldenRetrievalCase,
    RelevanceJudgment,
)
from app.application.retrieval.retrieval_experiment import (
    RetrievalEvaluationDataset,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _CorpusReader:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks
        self.call_count = 0

    def list_current_chunks(
        self,
        *,
        chunker_version: str,
        source_reference: str | None = None,
    ) -> list[EvidenceChunk]:
        self.call_count += 1
        return list(self.chunks)


class _Embedder:
    def __init__(self) -> None:
        self._spec = EmbeddingSpec(
            provider="test",
            model_name="fake",
            model_revision="v1",
            dimension=2,
        )
        self.document_calls = 0
        self.query_calls = 0

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.document_calls += 1
        return tuple(
            (1.0, 0.0) if "院校" in text else (0.0, 1.0)
            for text in texts
        )

    def embed_query(self, query: str) -> tuple[float, ...]:
        self.query_calls += 1
        return (1.0, 0.0)


class CurrentCorpusHybridEvaluatorTest(unittest.TestCase):
    def test_builds_each_retriever_once_and_evaluates_cases(self) -> None:
        relevant = self._chunk(1, "院校信息支持手动输入")
        unrelated = self._chunk(2, "面试通知")
        reader = _CorpusReader([relevant, unrelated])
        embedder = _Embedder()

        report = CurrentCorpusHybridEvaluator(
            reader,
            embedder,
        ).evaluate(
            self._dataset(relevant.evidence_id),
            top_k=1,
            candidate_k=2,
            allow_provisional=True,
        )

        self.assertEqual(1, reader.call_count)
        self.assertEqual(1, embedder.document_calls)
        self.assertEqual(2, embedder.query_calls)
        self.assertEqual(1.0, report.metrics.macro_recall_at_k)
        self.assertEqual(2, report.candidate_k)
        self.assertEqual(60, report.rank_constant)

    def test_rejects_candidate_k_smaller_than_output_k(self) -> None:
        relevant = self._chunk(1, "院校信息支持手动输入")

        with self.assertRaisesRegex(ValueError, "at least top_k"):
            CurrentCorpusHybridEvaluator(
                _CorpusReader([relevant]),
                _Embedder(),
            ).evaluate(
                self._dataset(relevant.evidence_id),
                top_k=5,
                candidate_k=3,
                allow_provisional=True,
            )

    def test_parameter_grid_reuses_document_and_query_vectors(self) -> None:
        relevant = self._chunk(1, "院校信息支持手动输入")
        unrelated = self._chunk(2, "面试通知")
        reader = _CorpusReader([relevant, unrelated])
        embedder = _Embedder()

        report = CurrentCorpusHybridEvaluator(
            reader,
            embedder,
        ).evaluate_parameter_grid(
            self._dataset(relevant.evidence_id),
            top_k=1,
            candidate_ks=(1, 2),
            rank_constants=(10, 60),
            allow_provisional=True,
        )

        self.assertEqual(1, reader.call_count)
        self.assertEqual(1, embedder.document_calls)
        self.assertEqual(2, embedder.query_calls)
        self.assertEqual(4, len(report.evaluations))

    @staticmethod
    def _dataset(judgment_id: str) -> RetrievalEvaluationDataset:
        cases = tuple(
            GoldenRetrievalCase(
                case_id=f"case-{index}",
                query="学校名单里没有我的大学怎么办？",
                judgments=(
                    RelevanceJudgment(
                        evidence_id=judgment_id,
                        relevance_grade=3,
                    ),
                ),
            )
            for index in (1, 2)
        )
        return RetrievalEvaluationDataset(
            dataset_id="dataset-v1",
            annotation_status="provisional",
            annotated_by="student",
            source_reference="official-source",
            document_sha256="a" * 64,
            chunker_version="structure-v1",
            tokenizer_version="mixed-cjk-bigram-v1",
            cases=cases,
        )

    @staticmethod
    def _chunk(index: int, text: str) -> EvidenceChunk:
        return EvidenceChunk(
            evidence_id="ev_" + f"{index:064x}",
            document_sha256="a" * 64,
            source_reference="official-source",
            source_fragment_ordinal=index,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text=text,
            location=EvidenceLocation(heading_path=("FAQ",)),
        )


if __name__ == "__main__":
    unittest.main()
