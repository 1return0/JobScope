from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.application.retrieval.retrieval_experiment import (
    LexicalFailureDiagnosticReport,
)
from app.bootstrap import build_current_corpus_bm25_evaluator
from app.config import load_settings
from app.infrastructure.retrieval.retrieval_evaluation_json import (
    load_retrieval_evaluation_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain BM25 failures using query tokens, lexical overlap "
            "and each gold Evidence's full ranking position."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/evaluation/huawei-campus-faq-retrieval-v1.json"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-provisional", action="store_true")
    return parser.parse_args()


def report_to_payload(
    report: LexicalFailureDiagnosticReport,
) -> dict[str, Any]:
    return {
        "dataset_id": report.dataset_id,
        "annotation_status": report.annotation_status,
        "corpus_size": report.corpus_size,
        "top_k": report.top_k,
        "cases": [
            {
                "case_id": case.case_id,
                "query": case.query,
                "query_tokens": list(case.query_tokens),
                "classification": case.classification,
                "contributing_factors": list(
                    case.contributing_factors
                ),
                "gold_evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "relevance_grade": item.relevance_grade,
                        "matched_terms": list(item.matched_terms),
                        "full_ranking_position": (
                            item.full_ranking_position
                        ),
                        "status_at_k": item.status_at_k,
                    }
                    for item in case.gold_evidence
                ],
            }
            for case in report.cases
        ],
    }


def main() -> int:
    args = parse_args()
    dataset = load_retrieval_evaluation_dataset(args.dataset)
    evaluator = build_current_corpus_bm25_evaluator(
        load_settings()
    )
    report = evaluator.diagnose_lexical_failures(
        dataset,
        top_k=args.top_k,
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
