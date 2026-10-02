from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.jobs.job_agent_api import (
    JOB_AGENT_SESSION_COOKIE_NAME,
    build_job_agent_router,
)
from app.application.jobs.job_agent_sessions import IssuedJobAgentSession


class _Runtime:
    session_cookie_secure = False

    def __init__(self) -> None:
        self.issued = IssuedJobAgentSession(
            session_token="opaque-browser-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    def issue_browser_session(self) -> IssuedJobAgentSession:
        return self.issued

    def invoke(self, _user_query: str):
        raise AssertionError("not used in this test")


class JobAgentSessionApiTest(unittest.TestCase):
    def test_issues_http_only_cookie_without_exposing_owner_id(self) -> None:
        app = FastAPI()
        app.include_router(build_job_agent_router(_Runtime()))

        response = TestClient(app).post("/v1/job-agent/sessions", json={})

        self.assertEqual(201, response.status_code)
        self.assertEqual(["expires_at"], list(response.json()))
        cookie = response.headers["set-cookie"]
        self.assertIn(f"{JOB_AGENT_SESSION_COOKIE_NAME}=opaque-browser-token", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("Path=/v1/job-agent", cookie)

    def test_returns_503_when_runtime_is_not_prepared(self) -> None:
        app = FastAPI()
        app.include_router(build_job_agent_router())

        response = TestClient(app).post("/v1/job-agent/sessions", json={})

        self.assertEqual(503, response.status_code)
