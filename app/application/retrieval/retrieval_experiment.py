from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.application.retrieval.current_corpus_search import CurrentCorpusReader
from app.application.retrieval.lexical_retrieval import (
    Bm25Config,
    Bm25Retriever,
    tokenize_lexical,
)
from app.application.retrieval.retrieval_evaluation import (
    GoldenRetrievalCase,
    RetrievalEvaluationReport,
    RetrievalEvaluator,
)
from app.domain.documents.document_chunking import EvidenceChunk


AnnotationStatus = Literal["provisional", "reviewed"]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationDataset:
    dataset_id: str
    annotation_status: AnnotationStatus
    annotated_by: str
    source_reference: str
    document_sha256: str
    chunker_version: str
    tokenizer_version: str
    cases: tuple[GoldenRetrievalCase, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "annotated_by",
            "source_reference",
            "chunker_version",
            "tokenizer_version",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.annotation_status not in ("provisional", "reviewed"):
            raise ValueError("unsupported annotation_status")
        if len(self.document_sha256) != 64:
            raise ValueError("document_sha256 must be a SHA-256 digest")
        try:
            int(self.document_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "document_sha256 must be a SHA-256 digest"
            ) from error
        if not self.cases:
            raise ValueError("evaluation dataset must contain cases")


@dataclass(frozen=True, slots=True)
class RetrievalExperimentReport:
    dataset_id: str
    annotation_status: AnnotationStatus
    corpus_size: int
    chunker_version: str
    tokenizer_version: str
    metrics: RetrievalEvaluationReport


@dataclass(frozen=True, slots=True)
class RetrievalCaseFailureAnalysis:
    case_id: str
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalCutoffSweepReport:
    dataset_id: str
    annotation_status: AnnotationStatus
    corpus_size: int
    chunker_version: str
    tokenizer_version: str
    evaluations: tuple[RetrievalEvaluationReport, ...]
    max_cutoff_failures: tuple[
        RetrievalCaseFailureAnalysis,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class GoldEvidenceLexicalDiagnostic:
    evidence_id: str
    relevance_grade: int
    matched_terms: tuple[str, ...]
    full_ranking_position: int | None
    status_at_k: str


@dataclass(frozen=True, slots=True)
class RetrievalCaseLexicalDiagnostic:
    case_id: str
    query: str
    query_tokens: tuple[str, ...]
    classification: str
    contributing_factors: tuple[str, ...]
    gold_evidence: tuple[GoldEvidenceLexicalDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class LexicalFailureDiagnosticReport:
    dataset_id: str
    annotation_status: AnnotationStatus
    corpus_size: int
    top_k: int
    cases: tuple[RetrievalCaseLexicalDiagnostic, ...]


class CurrentCorpusBm25Evaluator:
    def __init__(
        self,
        corpus_reader: CurrentCorpusReader,
        *,
        bm25_config: Bm25Config = Bm25Config(),
    ) -> None:
        self._corpus_reader = corpus_reader
        self._bm25_config = bm25_config
        self._evaluator = RetrievalEvaluator()

    def evaluate(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        top_k: int,
        allow_provisional: bool = False,
    ) -> RetrievalExperimentReport:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        retriever = self._prepare_retriever(
            dataset,
            allow_provisional=allow_provisional,
        )
        metrics = self._evaluate_with_retriever(
            retriever,
            dataset,
            top_k=top_k,
        )
        return RetrievalExperimentReport(
            dataset_id=dataset.dataset_id,
            annotation_status=dataset.annotation_status,
            corpus_size=retriever.corpus_size,
            chunker_version=dataset.chunker_version,
            tokenizer_version=retriever.tokenizer_version,
            metrics=metrics,
        )

    def evaluate_cutoffs(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        top_ks: tuple[int, ...],
        allow_provisional: bool = False,
    ) -> RetrievalCutoffSweepReport:
        if not top_ks:
            raise ValueError("top_ks must not be empty")
        if any(top_k < 1 for top_k in top_ks):
            raise ValueError("every top_k must be positive")
        if len(top_ks) != len(set(top_ks)):
            raise ValueError("top_ks must not contain duplicates")

        normalized_top_ks = tuple(sorted(top_ks))
        retriever = self._prepare_retriever(
            dataset,
            allow_provisional=allow_provisional,
        )
        evaluations = tuple(
            self._evaluate_with_retriever(
                retriever,
                dataset,
                top_k=top_k,
            )
            for top_k in normalized_top_ks
        )
        return RetrievalCutoffSweepReport(
            dataset_id=dataset.dataset_id,
            annotation_status=dataset.annotation_status,
            corpus_size=retriever.corpus_size,
            chunker_version=dataset.chunker_version,
            tokenizer_version=retriever.tokenizer_version,
            evaluations=evaluations,
            max_cutoff_failures=self._analyze_failures(
                evaluations[-1]
            ),
        )

    def diagnose_lexical_failures(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        top_k: int,
        allow_provisional: bool = False,
    ) -> LexicalFailureDiagnosticReport:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        chunks = self._load_validated_chunks(
            dataset,
            allow_provisional=allow_provisional,
        )
        chunks_by_id = {
            chunk.evidence_id: chunk for chunk in chunks
        }
        retriever = Bm25Retriever(chunks, self._bm25_config)
        cases = tuple(
            self._diagnose_case(
                case,
                chunks_by_id=chunks_by_id,
                retriever=retriever,
                top_k=top_k,
            )
            for case in dataset.cases
        )
        return LexicalFailureDiagnosticReport(
            dataset_id=dataset.dataset_id,
            annotation_status=dataset.annotation_status,
            corpus_size=retriever.corpus_size,
            top_k=top_k,
            cases=cases,
        )

    def _prepare_retriever(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        allow_provisional: bool,
    ) -> Bm25Retriever:
        chunks = self._load_validated_chunks(
            dataset,
            allow_provisional=allow_provisional,
        )
        return Bm25Retriever(chunks, self._bm25_config)

    def _load_validated_chunks(
        self,
        dataset: RetrievalEvaluationDataset,
        *,
        allow_provisional: bool,
    ) -> list[EvidenceChunk]:
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

        return chunks

    def _evaluate_with_retriever(
        self,
        retriever: Bm25Retriever,
        dataset: RetrievalEvaluationDataset,
        *,
        top_k: int,
    ) -> RetrievalEvaluationReport:
        return self._evaluator.evaluate(
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

    @staticmethod
    def _diagnose_case(
        case: GoldenRetrievalCase,
        *,
        chunks_by_id: dict[str, EvidenceChunk],
        retriever: Bm25Retriever,
        top_k: int,
    ) -> RetrievalCaseLexicalDiagnostic:
        query_tokens = tokenize_lexical(case.query)
        query_token_set = set(query_tokens)
        full_results = retriever.search(
            case.query,
            top_k=retriever.corpus_size,
        )
        rank_by_evidence_id = {
            result.chunk.evidence_id: result.rank
            for result in full_results
        }

        gold_evidence: list[GoldEvidenceLexicalDiagnostic] = []
        for judgment in case.judgments:
            chunk = chunks_by_id[judgment.evidence_id]
            matched_terms = tuple(
                sorted(
                    query_token_set
                    & set(tokenize_lexical(chunk.text))
                )
            )
            full_rank = rank_by_evidence_id.get(
                judgment.evidence_id
            )
            if not matched_terms:
                status_at_k = "no-lexical-overlap"
            elif full_rank is not None and full_rank <= top_k:
                status_at_k = "retrieved"
            else:
                status_at_k = "ranked-below-cutoff"
            gold_evidence.append(
                GoldEvidenceLexicalDiagnostic(
                    evidence_id=judgment.evidence_id,
                    relevance_grade=judgment.relevance_grade,
                    matched_terms=matched_terms,
                    full_ranking_position=full_rank,
                    status_at_k=status_at_k,
                )
            )

        statuses = {
            item.status_at_k for item in gold_evidence
        }
        retrieved_ranks = [
            item.full_ranking_position
            for item in gold_evidence
            if item.status_at_k == "retrieved"
            and item.full_ranking_position is not None
        ]
        factors: list[str] = []
        if not query_tokens:
            factors.append("query-produced-no-tokens")
        else:
            if statuses == {"no-lexical-overlap"}:
                factors.append("no-gold-lexical-overlap")
            elif "no-lexical-overlap" in statuses:
                factors.append("partial-gold-lexical-overlap")
            if "ranked-below-cutoff" in statuses:
                factors.append("gold-ranked-below-cutoff")
            if retrieved_ranks and min(retrieved_ranks) > 1:
                factors.append("relevant-results-ranked-late")
        if not factors:
            factors.append("lexical-overlap-covered")

        return RetrievalCaseLexicalDiagnostic(
            case_id=case.case_id,
            query=case.query,
            query_tokens=query_tokens,
            classification=factors[0],
            contributing_factors=tuple(factors),
            gold_evidence=tuple(gold_evidence),
        )

    @staticmethod
    def _analyze_failures(
        report: RetrievalEvaluationReport,
    ) -> tuple[RetrievalCaseFailureAnalysis, ...]:
        analyses: list[RetrievalCaseFailureAnalysis] = []
        for case in report.cases:
            reasons: list[str] = []
            if case.first_relevant_rank is None:
                reasons.append("no-relevant-evidence-retrieved")
            elif case.first_relevant_rank > 1:
                reasons.append("first-relevant-result-ranked-late")
            if case.recall_at_k < 1:
                reasons.append("incomplete-recall")
            if case.ndcg_at_k < 1:
                reasons.append("suboptimal-relevance-order")
            if reasons:
                analyses.append(
                    RetrievalCaseFailureAnalysis(
                        case_id=case.case_id,
                        failure_reasons=tuple(reasons),
                    )
                )
        return tuple(analyses)
