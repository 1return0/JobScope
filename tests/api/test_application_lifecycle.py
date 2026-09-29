import unittest

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.application_lifecycle import build_application_lifespan


class _Runtime:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _unused_retrieval_factory():
    raise AssertionError("retrieval factory must not be called")


class ApplicationLifecycleTest(unittest.TestCase):
    def test_creates_one_job_agent_runtime_and_closes_it(self) -> None:
        runtime = _Runtime()
        factory_calls = 0

        def runtime_factory(hybrid_service, answer_service):
            nonlocal factory_calls
            factory_calls += 1
            self.assertIsNone(hybrid_service)
            self.assertIsNone(answer_service)
            return runtime

        app = FastAPI(
            lifespan=build_application_lifespan(
                retrieval_enabled=False,
                retrieval_service_factory=_unused_retrieval_factory,
                job_agent_enabled=True,
                job_agent_runtime_factory=runtime_factory,
            )
        )

        @app.get("/probe")
        def probe(request: Request):
            return {
                "runtime_available": hasattr(
                    request.app.state,
                    "job_agent_runtime",
                )
            }

        with TestClient(app) as client:
            self.assertEqual(
                {"runtime_available": True},
                client.get("/probe").json(),
            )
            self.assertEqual(1, factory_calls)
            self.assertFalse(runtime.closed)

        self.assertTrue(runtime.closed)
        self.assertFalse(hasattr(app.state, "job_agent_runtime"))

    def test_disabled_job_agent_does_not_create_runtime(self) -> None:
        factory_calls = 0

        def runtime_factory(hybrid_service, answer_service):
            nonlocal factory_calls
            factory_calls += 1
            return _Runtime()

        app = FastAPI(
            lifespan=build_application_lifespan(
                retrieval_enabled=False,
                retrieval_service_factory=_unused_retrieval_factory,
                job_agent_enabled=False,
                job_agent_runtime_factory=runtime_factory,
            )
        )
        with TestClient(app):
            self.assertFalse(hasattr(app.state, "job_agent_runtime"))
        self.assertEqual(0, factory_calls)

    def test_enabled_job_agent_requires_runtime_factory(self) -> None:
        app = FastAPI(
            lifespan=build_application_lifespan(
                retrieval_enabled=False,
                retrieval_service_factory=_unused_retrieval_factory,
                job_agent_enabled=True,
            )
        )
        with self.assertRaisesRegex(ValueError, "factory"):
            with TestClient(app):
                pass


if __name__ == "__main__":
    unittest.main()
