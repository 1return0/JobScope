import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.answering.answer_api import build_answer_router
from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerResult,
)
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchReport,
    HybridSearchResult,
)
from app.application.answering.grounded_answer_service import GroundedAnswerResult
from app.application.answering.grounded_answering import (
    CitationValidationReport,
    EvidenceCitation,
    GroundedAnswerDraft,
    GroundedClaim,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation
from app.infrastructure.llm.openai_compatible_answer_model import (
    AnswerModelResponseError,
)


class _AnswerService:
    def answer(self, question: str, *, top_k: int = 5):
        chunk = EvidenceChunk(
            evidence_id=f"ev_{'1' * 64}",
            document_sha256="a" * 64,
            source_reference="official-source",
            source_fragment_ordinal=0,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text="工作地点位于上海。",
            location=EvidenceLocation(
                page_number=2,
                heading_path=("岗位信息",),
            ),
        )
        retrieval = CurrentCorpusHybridSearchReport(
            query=question.strip(),
            top_k=1,
            candidate_k=10,
            rank_constant=60,
            corpus_size=60,
            chunker_version="structure-v1",
            tokenizer_version="mixed-cjk-bigram-v1",
            embedding_identity="embedding-v1",
            corpus_fingerprint="corpus-v1",
            results=(
                HybridSearchResult(
                    rank=1,
                    fusion_score=0.03,
                    ranks_by_retriever=(("dense", 1),),
                    chunk=chunk,
                ),
            ),
        )
        draft = GroundedAnswerDraft(
            status="answered",
            claims=(
                GroundedClaim(
                    text="工作地点是上海。",
                    citations=(
                        EvidenceCitation(
                            evidence_id=chunk.evidence_id,
                            quoted_text="工作地点位于上海",
                        ),
                    ),
                ),
            ),
        )
        return CurrentCorpusAnswerResult(
            retrieval=retrieval,
            answer=GroundedAnswerResult(
                draft=draft,
                validation=CitationValidationReport(
                    valid=True,
                    checked_claim_count=1,
                    checked_citation_count=1,
                    issues=(),
                ),
                model_called=True,
            ),
        )


class _InvalidModelResponseService:
    def answer(self, question: str, *, top_k: int = 5):
        raise AnswerModelResponseError("blank model content")


class AnswerApiTest(unittest.TestCase):
    def test_returns_answer_with_source_trace(self) -> None:
        app = FastAPI()
        app.include_router(build_answer_router(_AnswerService()))
        response = TestClient(app).post(
            "/v1/answers/grounded",
            json={"question": "工作地点在哪里？", "top_k": 5},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("answered", payload["status"])
        self.assertTrue(payload["grounding_valid"])
        citation = payload["claims"][0]["citations"][0]
        self.assertEqual("official-source", citation["source_reference"])
        self.assertEqual(2, citation["page_number"])
        self.assertEqual("corpus-v1", payload["retrieval"]["corpus_fingerprint"])

    def test_returns_503_when_service_is_unavailable(self) -> None:
        app = FastAPI()
        app.include_router(build_answer_router())

        response = TestClient(app).post(
            "/v1/answers/grounded",
            json={"question": "工作地点在哪里？"},
        )

        self.assertEqual(503, response.status_code)

    def test_maps_invalid_model_response_to_safe_502(self) -> None:
        app = FastAPI()
        app.include_router(
            build_answer_router(_InvalidModelResponseService())
        )

        response = TestClient(app).post(
            "/v1/answers/grounded",
            json={"question": "工作地点在哪里？"},
        )

        self.assertEqual(502, response.status_code)
        self.assertEqual(
            "answer model returned an invalid grounded response",
            response.json()["detail"],
        )


if __name__ == "__main__":
    unittest.main()
