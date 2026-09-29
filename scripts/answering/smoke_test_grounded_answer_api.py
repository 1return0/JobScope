from __future__ import annotations

import argparse
import json
from time import perf_counter

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.answering.answer_api import build_answer_router
from app.api.retrieval.retrieval_lifecycle import build_retrieval_lifespan
from app.config import load_settings
from app.runtime_composition import (
    compose_answer_service,
    compose_hybrid_search_service,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real grounded-answer API dependency chain."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="学校名单里没有我的大学怎么办？",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    settings = load_settings(dotenv_override=True)
    if not settings.hybrid_retrieval_enabled:
        raise RuntimeError(
            "set JOBSCOPE_HYBRID_RETRIEVAL_ENABLED=true"
        )
    if not settings.answer_generation_enabled:
        raise RuntimeError(
            "set JOBSCOPE_ANSWER_GENERATION_ENABLED=true"
        )

    app = FastAPI(
        lifespan=build_retrieval_lifespan(
            enabled=True,
            service_factory=lambda: compose_hybrid_search_service(
                settings
            ),
            answer_enabled=True,
            answer_service_factory=lambda hybrid: compose_answer_service(
                settings,
                hybrid,
            ),
        )
    )
    app.include_router(build_answer_router())

    startup_started = perf_counter()
    with TestClient(app) as client:
        startup_seconds = perf_counter() - startup_started
        query_started = perf_counter()
        response = client.post(
            "/v1/answers/grounded",
            json={
                "question": arguments.query,
                "top_k": arguments.top_k,
            },
        )
        query_seconds = perf_counter() - query_started
        response.raise_for_status()
        payload = response.json()

    print(
        json.dumps(
            {
                "startup_seconds": round(startup_seconds, 6),
                "query_seconds": round(query_seconds, 6),
                "status": payload["status"],
                "model_called": payload["model_called"],
                "grounding_valid": payload["grounding_valid"],
                "claim_count": len(payload["claims"]),
                "claims": payload["claims"],
                "corpus_fingerprint": payload["retrieval"][
                    "corpus_fingerprint"
                ],
                "embedding_identity": payload["retrieval"][
                    "embedding_identity"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
