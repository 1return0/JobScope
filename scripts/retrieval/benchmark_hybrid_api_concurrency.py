from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        description="Run a bounded concurrent Hybrid API experiment."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--service-capacity", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must not be negative")
    if args.requests < 1:
        raise ValueError("requests must be positive")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.service_capacity < 1:
        raise ValueError("service_capacity must be positive")

    app = FastAPI(
        lifespan=build_retrieval_lifespan(
            enabled=True,
            service_factory=lambda: build_hybrid_search_service(
                max_concurrent_queries=args.service_capacity
            ),
        )
    )
    app.include_router(build_retrieval_router())
    payload = {"query": args.query, "top_k": args.top_k}

    latencies: list[float] = []
    status_codes: list[int] = []
    with TestClient(app) as client:
        for _ in range(args.warmup):
            response = client.post("/v1/retrieval/hybrid", json=payload)
            response.raise_for_status()

        def send_one() -> tuple[float, int]:
            started = perf_counter()
            response = client.post(
                "/v1/retrieval/hybrid",
                json=payload,
            )
            elapsed = perf_counter() - started
            return elapsed, response.status_code

        wall_started = perf_counter()
        with ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [
                executor.submit(send_one)
                for _ in range(args.requests)
            ]
            for future in as_completed(futures):
                elapsed, status_code = future.result()
                latencies.append(elapsed)
                status_codes.append(status_code)
        wall_seconds = perf_counter() - wall_started

    success_count = sum(code == 200 for code in status_codes)
    output = {
        "mode": "concurrent-testclient",
        "concurrency": args.concurrency,
        "service_capacity": args.service_capacity,
        "request_count": args.requests,
        "success_count": success_count,
        "error_count": args.requests - success_count,
        "wall_seconds": round(wall_seconds, 6),
        "throughput_requests_per_second": round(
            args.requests / wall_seconds,
            6,
        ),
        "latency_seconds": {
            "p50": round(nearest_rank_percentile(latencies, 0.50), 6),
            "p95": round(nearest_rank_percentile(latencies, 0.95), 6),
            "p99": round(nearest_rank_percentile(latencies, 0.99), 6),
            "min": round(min(latencies), 6),
            "max": round(max(latencies), 6),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if success_count == args.requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
