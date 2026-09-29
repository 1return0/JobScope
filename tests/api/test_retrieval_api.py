import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.retrieval.retrieval_api import build_retrieval_router
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchReport,
    HybridSearchCapacityError,
    HybridSearchResult,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _PreparedService:
    def search(self, query: str, *, top_k: int = 5):
        chunk = EvidenceChunk(
            evidence_id="ev_" + "1" * 64,
            document_sha256="a" * 64,
            source_reference="official-source",
            source_fragment_ordinal=1,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text="手动输入标准院校名称。",
            location=EvidenceLocation(heading_path=("院校信息",)),
        )
        return CurrentCorpusHybridSearchReport(
            query=query.strip(),
            top_k=1,
            candidate_k=10,
            rank_constant=60,
            corpus_size=60,
            chunker_version="structure-v1",
            tokenizer_version="mixed-cjk-bigram-v1",
            embedding_identity="emb_test",
            corpus_fingerprint="corpus_test",
            results=(
                HybridSearchResult(
                    rank=1,
                    fusion_score=0.03,
                    ranks_by_retriever=(
                        ("bm25", 1),
                        ("dense", 2),
                    ),
                    chunk=chunk,
                ),
            ),
        )


class _UnpreparedService:
    def search(self, query: str, *, top_k: int = 5):
        raise RuntimeError("hybrid search service is not prepared")


class _OverloadedService:
    def search(self, query: str, *, top_k: int = 5):
        raise HybridSearchCapacityError("capacity exhausted")


class RetrievalApiTest(unittest.TestCase):
    def test_returns_auditable_hybrid_results(self) -> None:
        client = self._client(_PreparedService())

        response = client.post(
            "/v1/retrieval/hybrid",
            json={"query": "学校名单里没有我的大学", "top_k": 5},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(60, payload["corpus_size"])
        self.assertEqual("ev_" + "1" * 64, payload["results"][0]["evidence_id"])
        self.assertEqual(
            [
                {"retriever": "bm25", "rank": 1},
                {"retriever": "dense", "rank": 2},
            ],
            payload["results"][0]["ranks_by_retriever"],
        )

    def test_returns_503_until_service_is_prepared(self) -> None:
        client = self._client(_UnpreparedService())

        response = client.post(
            "/v1/retrieval/hybrid",
            json={"query": "学校名单里没有我的大学"},
        )

        self.assertEqual(503, response.status_code)
        self.assertIn("not prepared", response.json()["detail"])

    def test_returns_429_when_query_capacity_is_exhausted(self) -> None:
        client = self._client(_OverloadedService())

        response = client.post(
            "/v1/retrieval/hybrid",
            json={"query": "学校名单里没有我的大学"},
        )

        self.assertEqual(429, response.status_code)
        self.assertEqual("1", response.headers["Retry-After"])

    @staticmethod
    def _client(service) -> TestClient:
        app = FastAPI()
        app.include_router(build_retrieval_router(service))
        return TestClient(app)


if __name__ == "__main__":
    unittest.main()
