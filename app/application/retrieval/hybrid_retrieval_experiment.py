from __future__ import annotations

from dataclasses import dataclass

from app.application.retrieval.current_corpus_search import CurrentCorpusReader
from app.application.retrieval.dense_retrieval import (
    DenseIndexBuilder,
    InMemoryDenseRetriever,
    TextEmbedder,
)
from app.application.retrieval.lexical_retrieval import Bm25Config, Bm25Retriever
from app.application.retrieval.reciprocal_rank_fusion import (
    reciprocal_rank_fusion,
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
class HybridRetrievalExperimentReport:
    dataset_id: str
    annotation_status: AnnotationStatus
    corpus_size: int
    chunker_version: str
    tokenizer_version: str
    embedding_identity: str
    corpus_fingerprint: str
    candidate_k: int
    rank_constant: int
    metrics: RetrievalEvaluationReport


@dataclass(frozen=True, slots=True)
class HybridParameterEvaluation:
    candidate_k: int
    rank_constant: int
    metrics: RetrievalEvaluationReport


@dataclass(frozen=True, slots=True)
class HybridParameterSweepReport:
    dataset_id: str
    annotation_status: AnnotationStatus
    corpus_size: int
    chunker_version: str
    tokenizer_version: str
    embedding_identity: str
    corpus_fingerprint: str
    evaluations: tuple[HybridParameterEvaluation, ...]


class CurrentCorpusHybridEvaluator:
    def __init__(
        self,
        corpus_reader: CurrentCorpusReader,
        embedder: TextEmbedder,
        *,
        bm25_config: Bm25Config = Bm25Config(),
    ) -> None:
        self._corpus_reader = corpus_reader
        self._embedder = embedder
        self._bm25_config = bm25_config
        self._evaluator = RetrievalEvaluator()

    def evaluate(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        top_k: int,
        candidate_k: int = 20,
        rank_constant: int = 60,
        allow_provisional: bool = False,
    ) -> HybridRetrievalExperimentReport:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")
        if rank_constant < 1:
            raise ValueError("rank_constant must be positive")
        if (
            dataset.annotation_status == "provisional"
            and not allow_provisional
        ):
            raise ValueError(
                "provisional evaluation requires explicit permission"
            )
        if dataset.tokenizer_version != (
            self._bm25_config.tokenizer_version
        ):
            raise ValueError(
                "dataset tokenizer_version does not match BM25 config"
            )

        chunks = self._corpus_reader.list_current_chunks(
            chunker_version=dataset.chunker_version,
            source_reference=dataset.source_reference,
        )
        self._validate_corpus(chunks, dataset)

        bm25 = Bm25Retriever(chunks, self._bm25_config)
        dense_index = DenseIndexBuilder(self._embedder).build(chunks)
        dense = InMemoryDenseRetriever(dense_index, self._embedder)

        def retrieve(query: str, requested_k: int) -> list[str]:
            bm25_ids = tuple(
                result.chunk.evidence_id
                for result in bm25.search(query, top_k=candidate_k)
            )
            dense_ids = tuple(
                result.chunk.evidence_id
                for result in dense.search(query, top_k=candidate_k)
            )
            fused = reciprocal_rank_fusion(
                {"bm25": bm25_ids, "dense": dense_ids},
                top_k=requested_k,
                rank_constant=rank_constant,
            )
            return [result.evidence_id for result in fused]

        metrics = self._evaluator.evaluate(
            list(dataset.cases),
            retrieve=retrieve,
            top_k=top_k,
        )
        return HybridRetrievalExperimentReport(
            dataset_id=dataset.dataset_id,
            annotation_status=dataset.annotation_status,
            corpus_size=dense.corpus_size,
            chunker_version=dense_index.chunker_version,
            tokenizer_version=bm25.tokenizer_version,
            embedding_identity=dense_index.embedding_identity,
            corpus_fingerprint=dense_index.corpus_fingerprint,
            candidate_k=candidate_k,
            rank_constant=rank_constant,
            metrics=metrics,
        )

    def evaluate_parameter_grid(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        top_k: int,
        candidate_ks: tuple[int, ...],
        rank_constants: tuple[int, ...],
        allow_provisional: bool = False,
    ) -> HybridParameterSweepReport:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not candidate_ks or not rank_constants:
            raise ValueError("parameter grid must not be empty")
        if len(candidate_ks) != len(set(candidate_ks)):
            raise ValueError("candidate_ks must not contain duplicates")
        if len(rank_constants) != len(set(rank_constants)):
            raise ValueError(
                "rank_constants must not contain duplicates"
            )
        if any(candidate_k < top_k for candidate_k in candidate_ks):
            raise ValueError("every candidate_k must be at least top_k")
        if any(value < 1 for value in rank_constants):
            raise ValueError("every rank_constant must be positive")
        if (
            dataset.annotation_status == "provisional"
            and not allow_provisional
        ):
            raise ValueError(
                "provisional evaluation requires explicit permission"
            )
        if dataset.tokenizer_version != (
            self._bm25_config.tokenizer_version
        ):
            raise ValueError(
                "dataset tokenizer_version does not match BM25 config"
            )

        chunks = self._corpus_reader.list_current_chunks(
            chunker_version=dataset.chunker_version,
            source_reference=dataset.source_reference,
        )
        self._validate_corpus(chunks, dataset)
        bm25 = Bm25Retriever(chunks, self._bm25_config)
        dense_index = DenseIndexBuilder(self._embedder).build(chunks)
        dense = InMemoryDenseRetriever(dense_index, self._embedder)

        max_candidate_k = max(candidate_ks)
        rankings_by_query = {
            case.query: {
                "bm25": tuple(
                    result.chunk.evidence_id
                    for result in bm25.search(
                        case.query,
                        top_k=max_candidate_k,
                    )
                ),
                "dense": tuple(
                    result.chunk.evidence_id
                    for result in dense.search(
                        case.query,
                        top_k=max_candidate_k,
                    )
                ),
            }
            for case in dataset.cases
        }

        evaluations: list[HybridParameterEvaluation] = []
        for candidate_k in sorted(candidate_ks):
            for rank_constant in sorted(rank_constants):
                metrics = self._evaluator.evaluate(
                    list(dataset.cases),
                    retrieve=lambda query, requested_k, candidate_k=(
                        candidate_k
                    ), rank_constant=rank_constant: [
                        result.evidence_id
                        for result in reciprocal_rank_fusion(
                            {
                                name: evidence_ids[:candidate_k]
                                for name, evidence_ids in (
                                    rankings_by_query[query].items()
                                )
                            },
                            top_k=requested_k,
                            rank_constant=rank_constant,
                        )
                    ],
                    top_k=top_k,
                )
                evaluations.append(
                    HybridParameterEvaluation(
                        candidate_k=candidate_k,
                        rank_constant=rank_constant,
                        metrics=metrics,
                    )
                )

        return HybridParameterSweepReport(
            dataset_id=dataset.dataset_id,
            annotation_status=dataset.annotation_status,
            corpus_size=dense.corpus_size,
            chunker_version=dense_index.chunker_version,
            tokenizer_version=bm25.tokenizer_version,
            embedding_identity=dense_index.embedding_identity,
            corpus_fingerprint=dense_index.corpus_fingerprint,
            evaluations=tuple(evaluations),
        )

    @staticmethod
    def _validate_corpus(
        chunks: list,
        dataset: RetrievalEvaluationDataset,
    ) -> None:
        if not chunks:
            raise ValueError("evaluation corpus must not be empty")
        if any(
            chunk.document_sha256 != dataset.document_sha256
            for chunk in chunks
        ):
            raise ValueError(
                "current corpus does not match dataset document hash"
            )
        corpus_ids = {chunk.evidence_id for chunk in chunks}
        judged_ids = {
            judgment.evidence_id
            for case in dataset.cases
            for judgment in case.judgments
        }
        missing_ids = sorted(judged_ids - corpus_ids)
        if missing_ids:
            raise ValueError(
                "gold judgments reference evidence outside current corpus: "
                + ", ".join(missing_ids)
            )
