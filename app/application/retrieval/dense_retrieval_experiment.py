from __future__ import annotations

from dataclasses import dataclass

from app.application.retrieval.current_corpus_search import CurrentCorpusReader
from app.application.retrieval.dense_retrieval import (
    DenseIndexBuilder,
    InMemoryDenseRetriever,
    TextEmbedder,
)
from app.application.retrieval.retrieval_evaluation import (
    RetrievalEvaluationReport,
    RetrievalEvaluator,
)
from app.application.retrieval.retrieval_experiment import (
    AnnotationStatus,
    RetrievalEvaluationDataset,
)


@dataclass(frozen=True, slots=True)
class DenseRetrievalExperimentReport:
    dataset_id: str
    annotation_status: AnnotationStatus
    corpus_size: int
    chunker_version: str
    embedding_identity: str
    corpus_fingerprint: str
    metrics: RetrievalEvaluationReport


class CurrentCorpusDenseEvaluator:
    def __init__(
        self,
        corpus_reader: CurrentCorpusReader,
        embedder: TextEmbedder,
    ) -> None:
        self._corpus_reader = corpus_reader
        self._embedder = embedder
        self._evaluator = RetrievalEvaluator()

    def evaluate(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        top_k: int,
        allow_provisional: bool = False,
    ) -> DenseRetrievalExperimentReport:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if (
            dataset.annotation_status == "provisional"
            and not allow_provisional
        ):
            raise ValueError(
                "provisional evaluation requires explicit permission"
            )

        chunks = self._corpus_reader.list_current_chunks(
            chunker_version=dataset.chunker_version,
            source_reference=dataset.source_reference,
        )
        if not chunks:
            raise ValueError("evaluation corpus must not be empty")
        if any(
            chunk.document_sha256 != dataset.document_sha256
            for chunk in chunks
        ):
            raise ValueError(
                "current corpus does not match dataset document hash"
            )

        corpus_evidence_ids = {
            chunk.evidence_id for chunk in chunks
        }
        judged_evidence_ids = {
            judgment.evidence_id
            for case in dataset.cases
            for judgment in case.judgments
        }
        missing_ids = sorted(
            judged_evidence_ids - corpus_evidence_ids
        )
        if missing_ids:
            raise ValueError(
                "gold judgments reference evidence outside current corpus: "
                + ", ".join(missing_ids)
            )

        index = DenseIndexBuilder(self._embedder).build(chunks)
        retriever = InMemoryDenseRetriever(index, self._embedder)
        metrics = self._evaluator.evaluate(
            list(dataset.cases),
            retrieve=lambda query, requested_k: [
                result.chunk.evidence_id
                for result in retriever.search(
                    query,
                    top_k=requested_k,
                )
            ],
            top_k=top_k,
        )
        return DenseRetrievalExperimentReport(
            dataset_id=dataset.dataset_id,
            annotation_status=dataset.annotation_status,
            corpus_size=retriever.corpus_size,
            chunker_version=index.chunker_version,
            embedding_identity=index.embedding_identity,
            corpus_fingerprint=index.corpus_fingerprint,
            metrics=metrics,
        )
