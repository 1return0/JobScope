import unittest

from app.application.retrieval.retrieval_evaluation import (
    GoldenRetrievalCase,
    RelevanceJudgment,
)
from app.application.retrieval.retrieval_experiment import (
    CurrentCorpusBm25Evaluator,
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


class CurrentCorpusBm25EvaluatorTest(unittest.TestCase):
    def test_runs_reviewed_dataset_against_current_corpus(self) -> None:
        chunk = self._chunk(1, "campus resume application")
        dataset = self._dataset(
            judgment_id=chunk.evidence_id,
            annotation_status="reviewed",
        )

        report = CurrentCorpusBm25Evaluator(
            _CorpusReader([chunk])
        ).evaluate(dataset, top_k=1)

        self.assertEqual(1, report.corpus_size)
        self.assertEqual(1.0, report.metrics.macro_recall_at_k)
        self.assertEqual(1.0, report.metrics.mean_reciprocal_rank)

    def test_requires_explicit_permission_for_provisional_data(self) -> None:
        chunk = self._chunk(1, "campus resume application")
        dataset = self._dataset(
            judgment_id=chunk.evidence_id,
            annotation_status="provisional",
        )

        with self.assertRaisesRegex(ValueError, "explicit permission"):
            CurrentCorpusBm25Evaluator(
                _CorpusReader([chunk])
            ).evaluate(dataset, top_k=1)

    def test_rejects_gold_id_outside_current_corpus(self) -> None:
        chunk = self._chunk(1, "campus resume application")
        dataset = self._dataset(
            judgment_id="ev_" + "2" * 64,
            annotation_status="reviewed",
        )

        with self.assertRaisesRegex(ValueError, "outside current corpus"):
            CurrentCorpusBm25Evaluator(
                _CorpusReader([chunk])
            ).evaluate(dataset, top_k=1)

    def test_cutoff_sweep_reads_corpus_once_and_analyzes_failure(
        self,
    ) -> None:
        irrelevant = self._chunk(1, "resume application")
        relevant = self._chunk(2, "resume application")
        reader = _CorpusReader([irrelevant, relevant])
        dataset = self._dataset(
            judgment_id=relevant.evidence_id,
            annotation_status="reviewed",
        )

        report = CurrentCorpusBm25Evaluator(reader).evaluate_cutoffs(
            dataset,
            top_ks=(3, 1),
        )

        self.assertEqual(1, reader.call_count)
        self.assertEqual(
            [1, 3],
            [item.top_k for item in report.evaluations],
        )
        self.assertEqual(
            "first-relevant-result-ranked-late",
            report.max_cutoff_failures[0].failure_reasons[0],
        )

    def test_cutoff_sweep_rejects_empty_or_duplicate_cutoffs(self) -> None:
        chunk = self._chunk(1, "resume application")
        evaluator = CurrentCorpusBm25Evaluator(
            _CorpusReader([chunk])
        )
        dataset = self._dataset(
            judgment_id=chunk.evidence_id,
            annotation_status="reviewed",
        )

        with self.assertRaisesRegex(ValueError, "empty"):
            evaluator.evaluate_cutoffs(dataset, top_ks=())
        with self.assertRaisesRegex(ValueError, "duplicates"):
            evaluator.evaluate_cutoffs(dataset, top_ks=(1, 1))

    def test_diagnoses_gold_without_lexical_overlap(self) -> None:
        chunk = self._chunk(1, "Java backend")
        dataset = self._dataset(
            judgment_id=chunk.evidence_id,
            annotation_status="reviewed",
        )

        report = CurrentCorpusBm25Evaluator(
            _CorpusReader([chunk])
        ).diagnose_lexical_failures(dataset, top_k=5)

        case = report.cases[0]
        self.assertEqual(
            "no-gold-lexical-overlap",
            case.classification,
        )
        self.assertEqual(
            "no-lexical-overlap",
            case.gold_evidence[0].status_at_k,
        )

    def test_diagnoses_gold_ranked_below_cutoff(self) -> None:
        irrelevant = self._chunk(1, "resume application")
        relevant = self._chunk(2, "resume application")
        dataset = self._dataset(
            judgment_id=relevant.evidence_id,
            annotation_status="reviewed",
        )

        report = CurrentCorpusBm25Evaluator(
            _CorpusReader([irrelevant, relevant])
        ).diagnose_lexical_failures(dataset, top_k=1)

        case = report.cases[0]
        self.assertEqual(
            "gold-ranked-below-cutoff",
            case.classification,
        )
        self.assertEqual(
            ("gold-ranked-below-cutoff",),
            case.contributing_factors,
        )
        self.assertEqual(2, case.gold_evidence[0].full_ranking_position)

    @staticmethod
    def _dataset(
        *,
        judgment_id: str,
        annotation_status: str,
    ) -> RetrievalEvaluationDataset:
        return RetrievalEvaluationDataset(
            dataset_id="dataset-v1",
            annotation_status=annotation_status,
            annotated_by="student",
            source_reference="official-source",
            document_sha256="a" * 64,
            chunker_version="structure-v1",
            tokenizer_version="mixed-cjk-bigram-v1",
            cases=(
                GoldenRetrievalCase(
                    case_id="case-1",
                    query="resume application",
                    judgments=(
                        RelevanceJudgment(
                            evidence_id=judgment_id,
                            relevance_grade=3,
                        ),
                    ),
                ),
            ),
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
