import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.jobs.job_agent_api import build_job_agent_router
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    JobAgentPlannerAuthenticationError,
    JobAgentPlannerRateLimitError,
    JobAgentPlannerUnavailableError,
    JobAgentPlannerUpstreamError,
)


class _Runtime:
    def __init__(self, state):
        self.state = state
        self.queries = []

    def invoke(self, user_query: str):
        self.queries.append(user_query)
        return self.state


class _FailingRuntime:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def invoke(self, _user_query: str):
        raise self.error


class JobAgentApiTest(unittest.TestCase):
    def test_maps_provider_failures_to_safe_http_responses(self) -> None:
        cases = (
            (JobAgentPlannerAuthenticationError("secret-key"), 503),
            (JobAgentPlannerUnavailableError("private-url"), 503),
            (JobAgentPlannerRateLimitError("provider-detail"), 429),
            (JobAgentPlannerUpstreamError("provider-request"), 502),
        )
        for planner_error, expected_status in cases:
            with self.subTest(planner_error=type(planner_error).__name__):
                app = FastAPI()
                app.include_router(
                    build_job_agent_router(_FailingRuntime(planner_error))
                )

                response = TestClient(app).post(
                    "/v1/job-agent/query",
                    json={"user_query": "Find Shanghai jobs"},
                )

                self.assertEqual(expected_status, response.status_code)
                self.assertNotIn(
                    str(planner_error),
                    response.json()["detail"],
                )

    def test_returns_503_when_runtime_is_not_prepared(self) -> None:
        app = FastAPI()
        app.include_router(build_job_agent_router())

        response = TestClient(app).post(
            "/v1/job-agent/query",
            json={"user_query": "Find Shanghai jobs"},
        )

        self.assertEqual(503, response.status_code)
        self.assertIn("disabled", response.json()["detail"])

    def test_returns_refusal_without_tool_data(self) -> None:
        runtime = _Runtime(
            {
                "planning_status": "refused",
                "planner_identity": "fake-planner-v1",
                "plan_reason": "The request is outside the job domain.",
                "tool_call": None,
                "tool_result": None,
                "tool_error": None,
            }
        )
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))

        response = TestClient(app).post(
            "/v1/job-agent/query",
            json={"user_query": "Predict lottery numbers"},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("refused", response.json()["planning_status"])
        self.assertIsNone(response.json()["tool_result"])

    def test_returns_tool_execution_result(self) -> None:
        runtime = _Runtime(
            {
                "planning_status": "tool_selected",
                "planner_identity": "fake-planner-v1",
                "plan_reason": "Current job data is required.",
                "tool_call": {
                    "call_id": "call-1",
                    "name": "search_current_jobs",
                    "arguments": {"location": "Shanghai", "limit": 5},
                },
                "tool_result": {
                    "call_id": "call-1",
                    "tool_name": "search_current_jobs",
                    "output": {
                        "status": "no_matches",
                        "result_count": 0,
                        "jobs": [],
                    },
                },
                "tool_error": None,
            }
        )
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))

        response = TestClient(app).post(
            "/v1/job-agent/query",
            json={"user_query": "Find Shanghai jobs"},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("tool_selected", payload["planning_status"])
        self.assertEqual(
            "search_current_jobs",
            payload["tool_result"]["tool_name"],
        )
        self.assertEqual(0, payload["tool_result"]["output"]["result_count"])

    def test_rejects_blank_query_before_runtime_call(self) -> None:
        runtime = _Runtime({})
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))

        response = TestClient(app).post(
            "/v1/job-agent/query",
            json={"user_query": ""},
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual([], runtime.queries)


if __name__ == "__main__":
    unittest.main()
