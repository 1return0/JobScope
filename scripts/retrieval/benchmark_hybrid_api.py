from __future__ import annotations

import argparse
import json
from statistics import fmean
from time import perf_counter

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.retrieval.retrieval_api import build_retrieval_router
from app.api.retrieval.retrieval_lifecycle import build_retrieval_lifespan
from app.application.retrieval.performance_metrics import (
    nearest_rank_percentile,
)
from app.main import build_hybrid_search_service


DEFAULT_QUERY = "学校名单里没有我的大学怎么办？"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequential warm Hybrid API latency benchmark."
        )
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must not be negative")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")

    app = FastAPI(
        lifespan=build_retrieval_lifespan(
            enabled=True,
            service_factory=build_hybrid_search_service,
        )
    )
    app.include_router(build_retrieval_router())
    request_payload = {"query": args.query, "top_k": args.top_k}

    startup_started = perf_counter()
    latencies: list[float] = []
    with TestClient(app) as client:
        startup_seconds = perf_counter() - startup_started
        for _ in range(args.warmup):
            response = client.post(
                "/v1/retrieval/hybrid",
                json=request_payload,
            )
            response.raise_for_status()
        for _ in range(args.iterations):
            started = perf_counter()
            response = client.post(
                "/v1/retrieval/hybrid",
                json=request_payload,
            )
            latencies.append(perf_counter() - started)
            response.raise_for_status()

    payload = {
        "mode": "sequential-testclient",
        "query": args.query,
        "top_k": args.top_k,
        "startup_seconds": round(startup_seconds, 6),
        "warmup_count": args.warmup,
        "measurement_count": args.iterations,
        "latency_seconds": {
            "mean": round(fmean(latencies), 6),
            "p50": round(nearest_rank_percentile(latencies, 0.50), 6),
            "p95": round(nearest_rank_percentile(latencies, 0.95), 6),
            "p99": round(nearest_rank_percentile(latencies, 0.99), 6),
            "min": round(min(latencies), 6),
            "max": round(max(latencies), 6),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
