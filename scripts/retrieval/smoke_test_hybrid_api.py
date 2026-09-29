from __future__ import annotations

import argparse
import json
from time import perf_counter

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.retrieval.retrieval_api import build_retrieval_router
from app.api.retrieval.retrieval_lifecycle import build_retrieval_lifespan
from app.main import build_hybrid_search_service


DEFAULT_QUERIES = (
    "学校名单里没有我的大学怎么办？",
    "什么时候通知面试结果？",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure real Hybrid API cold startup and warm query latency."
        )
    )
    parser.add_argument("query", nargs="*", default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = FastAPI(
        lifespan=build_retrieval_lifespan(
            enabled=True,
            service_factory=build_hybrid_search_service,
        )
    )
    app.include_router(build_retrieval_router())

    startup_started = perf_counter()
    query_reports: list[dict] = []
    with TestClient(app) as client:
        startup_seconds = perf_counter() - startup_started
        for query in args.query:
            query_started = perf_counter()
            response = client.post(
                "/v1/retrieval/hybrid",
                json={"query": query, "top_k": args.top_k},
            )
            query_seconds = perf_counter() - query_started
            response.raise_for_status()
            payload = response.json()
            query_reports.append(
                {
                    "query": query,
                    "status_code": response.status_code,
                    "latency_seconds": round(query_seconds, 6),
                    "result_count": len(payload["results"]),
                    "top_evidence_id": (
                        None
                        if not payload["results"]
                        else payload["results"][0]["evidence_id"]
                    ),
                    "corpus_fingerprint": payload[
                        "corpus_fingerprint"
                    ],
                    "embedding_identity": payload[
                        "embedding_identity"
                    ],
                }
            )

    print(
        json.dumps(
            {
                "startup_seconds": round(startup_seconds, 6),
                "query_count": len(query_reports),
                "queries": query_reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
