"""In-process HTTP acceptance coverage for JobScope's safe conversation boundary."""

from __future__ import annotations

from unittest import TestCase

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.api.jobs.job_agent_api import build_job_agent_router
from app.application.jobs.job_agent_planning import JobAgentPlan
from app.application.jobs.job_tool_graph import AgentToolCall
from app.config import Settings
from app.infrastructure.persistence.database import Base
from app.runtime_composition import compose_job_agent_runtime


class _Planner:
    identity = "job-agent-memory-http-acceptance-planner"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def plan(self, *, user_query: str, tools: object) -> JobAgentPlan:
        self.queries.append(user_query)
        return JobAgentPlan(
            decision="call_tool",
            reason="current job records are required",
            tool_call=AgentToolCall(
                call_id=f"acceptance-call-{len(self.queries)}",
                name="search_current_jobs",
                arguments={"limit": 1},
            ),
        )


class JobAgentMemoryHttpAcceptanceTest(TestCase):
    def setUp(self) -> None:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        self._planner = _Planner()
        self._runtime = compose_job_agent_runtime(
            Settings(),
            planner=self._planner,
            engine=engine,
            checkpointer=MemorySaver(),
        )
        app = FastAPI()
        app.include_router(build_job_agent_router(self._runtime))
        self._first_browser = TestClient(app)
        self._second_browser = TestClient(app)

    def tearDown(self) -> None:
        self._first_browser.close()
        self._second_browser.close()
        self._runtime.close()

    def test_http_session_ownership_recovers_only_its_own_safe_history(self) -> None:
        self._issue_session(self._first_browser)
        self._issue_session(self._second_browser)

        first_query = self._query(self._first_browser, "Find Shanghai internships")
        self._query(self._second_browser, "Find Beijing campus jobs")
        self._query(self._first_browser, "Only show campus recruitment")

        self.assertEqual(0, first_query.json()["harness"]["model_call_count"])
        self.assertEqual(1, first_query.json()["harness"]["tool_call_count"])
        self.assertEqual(
            "search_current_jobs",
            first_query.json()["harness"]["steps"][0]["operation_name"],
        )

        first_history = self._first_browser.get("/v1/job-agent/conversation")
        second_history = self._second_browser.get("/v1/job-agent/conversation")

        self.assertEqual(200, first_history.status_code)
        self.assertEqual(200, second_history.status_code)
        self.assertEqual(
            [
                {"role": "user", "content": "Find Shanghai internships"},
                {
                    "role": "assistant",
                    "content": (
                        "no_matches: registered read-only tool execution was completed"
                    ),
                },
                {"role": "user", "content": "Only show campus recruitment"},
                {
                    "role": "assistant",
                    "content": (
                        "no_matches: registered read-only tool execution was completed"
                    ),
                },
            ],
            first_history.json()["turns"],
        )
        self.assertEqual(
            [
                {"role": "user", "content": "Find Beijing campus jobs"},
                {
                    "role": "assistant",
                    "content": (
                        "no_matches: registered read-only tool execution was completed"
                    ),
                },
            ],
            second_history.json()["turns"],
        )

        second_request_for_first_browser = self._planner.queries[-1]
        self.assertIn("Find Shanghai internships", second_request_for_first_browser)
        self.assertNotIn("Find Beijing campus jobs", second_request_for_first_browser)

        self.assertEqual(
            204,
            self._first_browser.delete("/v1/job-agent/conversation").status_code,
        )
        self.assertEqual(
            [],
            self._first_browser.get("/v1/job-agent/conversation").json()["turns"],
        )
        self.assertEqual(
            2,
            len(self._second_browser.get("/v1/job-agent/conversation").json()["turns"]),
        )

    @staticmethod
    def _issue_session(browser: TestClient) -> None:
        response = browser.post("/v1/job-agent/sessions", json={})
        if response.status_code != 201:
            raise AssertionError(response.text)

    @staticmethod
    def _query(browser: TestClient, user_query: str):
        response = browser.post(
            "/v1/job-agent/query",
            json={"user_query": user_query},
        )
        if response.status_code != 200:
            raise AssertionError(response.text)
        return response
