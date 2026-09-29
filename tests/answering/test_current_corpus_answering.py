import json
import unittest

from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerService,
)
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchReport,
    HybridSearchResult,
)
from app.application.answering.grounded_answer_service import GroundedAnswerService
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _RecordingSearcher:
    def __init__(self, report: CurrentCorpusHybridSearchReport) -> None:
        self._report = report
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> CurrentCorpusHybridSearchReport:
        self.calls.append((query, top_k))
        return self._report


class _RecordingAnswerModel:
    def __init__(self, raw_output: str) -> None:
        self._raw_output = raw_output
        self.call_count = 0

    def generate(self, prompt: object) -> str:
        self.call_count += 1
        return self._raw_output


class CurrentCorpusAnswerServiceTest(unittest.TestCase):
    def test_preserves_hybrid_rank_order_for_answer_generation(self) -> None:
        first = self._chunk("1", "工作地点位于上海。")
        second = self._chunk("2", "该岗位专业不限。")
        searcher = _RecordingSearcher(self._report(first, second))
        model = _RecordingAnswerModel(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "工作地点是上海。",
                            "citations": [
                                {
                                    "evidence_id": first.evidence_id,
                                    "quoted_text": "工作地点位于上海",
                                }
                            ],
                        }
                    ],
                    "fallback_message": None,
                },
                ensure_ascii=False,
            )
        )
        service = CurrentCorpusAnswerService(
            searcher,
            GroundedAnswerService(model),
        )

        result = service.answer(" 工作地点在哪里？ ", top_k=2)

        self.assertEqual([("工作地点在哪里？", 2)], searcher.calls)
        self.assertEqual("answered", result.answer.draft.status)
        self.assertTrue(result.answer.validation.valid)
        self.assertEqual(first, result.retrieval.results[0].chunk)
        self.assertEqual(1, model.call_count)

    def test_empty_hybrid_results_refuse_without_model_call(self) -> None:
        searcher = _RecordingSearcher(self._report())
        model = _RecordingAnswerModel("must not be called")
        service = CurrentCorpusAnswerService(
            searcher,
            GroundedAnswerService(model),
        )

        result = service.answer("没有证据的问题")

        self.assertEqual(
            "insufficient_evidence",
            result.answer.draft.status,
        )
        self.assertFalse(result.answer.model_called)
        self.assertEqual(0, model.call_count)

    @staticmethod
    def _chunk(suffix: str, text: str) -> EvidenceChunk:
        return EvidenceChunk(
            evidence_id=f"ev_{suffix * 64}",
            document_sha256="a" * 64,
            source_reference="test-source",
            source_fragment_ordinal=0,
            chunk_ordinal=int(suffix) - 1,
            chunker_version="structure-v1",
            text=text,
            location=EvidenceLocation(),
        )

    @staticmethod
    def _report(
        *chunks: EvidenceChunk,
    ) -> CurrentCorpusHybridSearchReport:
        return CurrentCorpusHybridSearchReport(
            query="工作地点在哪里？",
            top_k=len(chunks),
            candidate_k=10,
            rank_constant=60,
            corpus_size=len(chunks),
            chunker_version="structure-v1",
            tokenizer_version="mixed-cjk-bigram-v1",
            embedding_identity="test-embedding",
            corpus_fingerprint="test-corpus",
            results=tuple(
                HybridSearchResult(
                    rank=index,
                    fusion_score=1 / (60 + index),
                    ranks_by_retriever=(("bm25", index),),
                    chunk=chunk,
                )
                for index, chunk in enumerate(chunks, start=1)
            ),
        )


if __name__ == "__main__":
    unittest.main()
