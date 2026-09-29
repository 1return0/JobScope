import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.retrieval.retrieval_api import build_retrieval_router
from app.api.answering.answer_api import build_answer_router
from app.api.retrieval.retrieval_lifecycle import build_retrieval_lifespan
from tests.api.test_retrieval_api import _PreparedService
from tests.api.test_answer_api import _AnswerService


class _LifecycleService(_PreparedService):
    def __init__(self) -> None:
        self.prepare_count = 0

    def prepare(self) -> None:
        self.prepare_count += 1


class RetrievalLifecycleTest(unittest.TestCase):
    def test_prepares_once_before_accepting_requests(self) -> None:
        service = _LifecycleService()
        app = FastAPI(
            lifespan=build_retrieval_lifespan(
                enabled=True,
                service_factory=lambda: service,
            )
        )
        app.include_router(build_retrieval_router())

        with TestClient(app) as client:
            first = client.post(
                "/v1/retrieval/hybrid",
                json={"query": "院校怎么填写", "top_k": 1},
            )
            second = client.post(
                "/v1/retrieval/hybrid",
                json={"query": "大学不在列表", "top_k": 1},
            )

        self.assertEqual(1, service.prepare_count)
        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)

    def test_disabled_lifecycle_returns_503_without_factory_call(self) -> None:
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return _LifecycleService()

        app = FastAPI(
            lifespan=build_retrieval_lifespan(
                enabled=False,
                service_factory=factory,
            )
        )
        app.include_router(build_retrieval_router())

        with TestClient(app) as client:
            response = client.post(
                "/v1/retrieval/hybrid",
                json={"query": "院校怎么填写", "top_k": 1},
            )

        self.assertEqual(0, factory_calls)
        self.assertEqual(503, response.status_code)

    def test_answer_service_reuses_prepared_hybrid_service(self) -> None:
        hybrid_service = _LifecycleService()
        received_hybrid_services = []

        def build_answer_service(active_hybrid_service):
            received_hybrid_services.append(active_hybrid_service)
            return _AnswerService()

        app = FastAPI(
            lifespan=build_retrieval_lifespan(
                enabled=True,
                service_factory=lambda: hybrid_service,
                answer_enabled=True,
                answer_service_factory=build_answer_service,
            )
        )
        app.include_router(build_answer_router())

        with TestClient(app) as client:
            response = client.post(
                "/v1/answers/grounded",
                json={"question": "工作地点在哪里？"},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual([hybrid_service], received_hybrid_services)
        self.assertEqual(1, hybrid_service.prepare_count)

    def test_answer_generation_requires_hybrid_retrieval(self) -> None:
        app = FastAPI(
            lifespan=build_retrieval_lifespan(
                enabled=False,
                service_factory=_LifecycleService,
                answer_enabled=True,
                answer_service_factory=lambda service: _AnswerService(),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires hybrid retrieval",
        ):
            with TestClient(app):
                pass


if __name__ == "__main__":
    unittest.main()
