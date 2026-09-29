from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.application.retrieval.retrieval_experiment import (
    RetrievalCutoffSweepReport,
)
from app.bootstrap import build_current_corpus_bm25_evaluator
from app.config import load_settings
from app.infrastructure.retrieval.retrieval_evaluation_json import (
    load_retrieval_evaluation_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BM25 retrieval quality at several Top-K cutoffs "
            "using one frozen in-memory view of the current corpus."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/evaluation/huawei-campus-faq-retrieval-v1.json"
        ),
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=(1, 3, 5),
    )
    parser.add_argument("--allow-provisional", action="store_true")
    return parser.parse_args()


def report_to_payload(
    report: RetrievalCutoffSweepReport,
) -> dict[str, Any]:
    return {
        "dataset_id": report.dataset_id,
        "annotation_status": report.annotation_status,
        "corpus_size": report.corpus_size,
        "chunker_version": report.chunker_version,
        "tokenizer_version": report.tokenizer_version,
        "cutoffs": [
            {
                "top_k": evaluation.top_k,
                "macro_recall_at_k": round(
                    evaluation.macro_recall_at_k,
                    6,
                ),
                "mean_reciprocal_rank": round(
                    evaluation.mean_reciprocal_rank,
                    6,
                ),
                "macro_ndcg_at_k": round(
                    evaluation.macro_ndcg_at_k,
                    6,
                ),
                "cases": [
                    {
                        "case_id": case.case_id,
                        "recall_at_k": round(
                            case.recall_at_k,
                            6,
                        ),
                        "reciprocal_rank": round(
                            case.reciprocal_rank,
                            6,
                        ),
                        "ndcg_at_k": round(
                            case.ndcg_at_k,
                            6,
                        ),
                        "first_relevant_rank": (
                            case.first_relevant_rank
                        ),
                    }
                    for case in evaluation.cases
                ],
            }
            for evaluation in report.evaluations
        ],
        "max_cutoff_failures": [
            {
                "case_id": analysis.case_id,
                "failure_reasons": list(
                    analysis.failure_reasons
                ),
            }
            for analysis in report.max_cutoff_failures
        ],
    }


def main() -> int:
    args = parse_args()
    dataset = load_retrieval_evaluation_dataset(args.dataset)
    evaluator = build_current_corpus_bm25_evaluator(
        load_settings()
    )
    report = evaluator.evaluate_cutoffs(
        dataset,
        top_ks=tuple(args.top_k_values),
        allow_provisional=args.allow_provisional,
    )
    print(
        json.dumps(
            report_to_payload(report),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
