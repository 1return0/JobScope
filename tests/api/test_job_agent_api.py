import unittest
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.jobs.job_agent_api import build_job_agent_router
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    JobAgentPlannerAuthenticationError,
    JobAgentPlannerRateLimitError,
    JobAgentPlannerUnavailableError,
    JobAgentPlannerUpstreamError,
)
from app.application.memory.long_term_memory import LongTermMemoryEntry


class _Runtime:
    def __init__(self, state):
        self.state = state
        self.queries = []
        self.owner_ids = []
        self.preferences: dict[tuple[str, str], LongTermMemoryEntry] = {}
        self.conversations: dict[str, tuple[str, ...]] = {}

    def resolve_browser_owner(self, session_token: str | None) -> str:
        owners = {
            "valid-session": "trusted-owner",
            "owner-a-session": "owner-a",
            "owner-b-session": "owner-b",
        }
        if session_token not in owners:
            from app.application.jobs.job_agent_sessions import (
                JobAgentSessionNotFoundError,
            )
            raise JobAgentSessionNotFoundError("invalid")
        return owners[session_token]

    def invoke(self, user_query: str, *, owner_id: str):
        self.queries.append(user_query)
        self.owner_ids.append(owner_id)
        return self.state

    def remember_preference(self, *, owner_id: str, preference_key: str, preference_value: str, user_consented: bool) -> LongTermMemoryEntry:
        if not user_consented:
            raise ValueError("consent is required")
        previous = self.preferences.get((owner_id, preference_key))
        now = datetime.now(timezone.utc)
        entry = LongTermMemoryEntry(owner_id, preference_key, preference_value, now, previous.created_at if previous else now, now, previous.revision + 1 if previous else 1)
        self.preferences[(owner_id, preference_key)] = entry
        return entry

    def list_preferences(self, *, owner_id: str):
        return tuple(entry for (entry_owner, _), entry in self.preferences.items() if entry_owner == owner_id)

    def forget_preference(self, *, owner_id: str, preference_key: str) -> bool:
        return self.preferences.pop((owner_id, preference_key), None) is not None

    def forget_all_preferences(self, *, owner_id: str) -> int:
        keys = [key for key in self.preferences if key[0] == owner_id]
        for key in keys:
            del self.preferences[key]
        return len(keys)

    def get_conversation_turns(self, *, owner_id: str) -> tuple[dict[str, str], ...]:
        return self.conversations.get(owner_id, ())

    def clear_conversation_turns(self, *, owner_id: str) -> bool:
        return self.conversations.pop(owner_id, None) is not None


class _FailingRuntime:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def resolve_browser_owner(self, _session_token: str | None) -> str:
        return "trusted-owner"

    def invoke(self, _user_query: str, *, owner_id: str):
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
                    cookies={"jobscope_job_agent_session": "valid-session"},
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
            cookies={"jobscope_job_agent_session": "valid-session"},
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
            cookies={"jobscope_job_agent_session": "valid-session"},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("tool_selected", payload["planning_status"])
        self.assertEqual(
            "search_current_jobs",
            payload["tool_result"]["tool_name"],
        )
        self.assertEqual(0, payload["tool_result"]["output"]["result_count"])
        self.assertEqual(["trusted-owner"], runtime.owner_ids)

    def test_rejects_blank_query_before_runtime_call(self) -> None:
        runtime = _Runtime({})
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))

        response = TestClient(app).post(
            "/v1/job-agent/query",
            json={"user_query": ""},
            cookies={"jobscope_job_agent_session": "valid-session"},
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual([], runtime.queries)

    def test_rejects_missing_or_invented_session_before_runtime_call(self) -> None:
        runtime = _Runtime({})
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))
        client = TestClient(app)

        for cookies in ({}, {"jobscope_job_agent_session": "invented"}):
            with self.subTest(cookies=cookies):
                response = client.post(
                    "/v1/job-agent/query",
                    json={"user_query": "Find Shanghai jobs"},
                    cookies=cookies,
                )
                self.assertEqual(401, response.status_code)
        self.assertEqual([], runtime.queries)

    def test_preference_api_is_session_scoped_and_can_forget(self) -> None:
        runtime = _Runtime({})
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))
        client = TestClient(app)
        cookies = {"jobscope_job_agent_session": "valid-session"}

        saved = client.put(
            "/v1/job-agent/memory/preferences",
            json={
                "preference_key": "preferred_location",
                "preference_value": "Shanghai",
                "user_consented": True,
            },
            cookies=cookies,
        )
        listed = client.get("/v1/job-agent/memory/preferences", cookies=cookies)
        deleted = client.delete(
            "/v1/job-agent/memory/preferences/preferred_location",
            cookies=cookies,
        )
        listed_after_delete = client.get(
            "/v1/job-agent/memory/preferences", cookies=cookies
        )

        self.assertEqual(200, saved.status_code)
        self.assertEqual("Shanghai", saved.json()["preference_value"])
        self.assertEqual(1, len(listed.json()["preferences"]))
        self.assertEqual(204, deleted.status_code)
        self.assertEqual([], listed_after_delete.json()["preferences"])

    def test_preference_api_does_not_leak_across_browser_owners(self) -> None:
        runtime = _Runtime({})
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))
        client = TestClient(app)

        saved = client.put(
            "/v1/job-agent/memory/preferences",
            json={
                "preference_key": "preferred_location",
                "preference_value": "Shanghai",
                "user_consented": True,
            },
            cookies={"jobscope_job_agent_session": "owner-a-session"},
        )
        other_owner = client.get(
            "/v1/job-agent/memory/preferences",
            cookies={"jobscope_job_agent_session": "owner-b-session"},
        )

        self.assertEqual(200, saved.status_code)
        self.assertEqual([], other_owner.json()["preferences"])

    def test_conversation_api_reads_and_clears_only_current_owner(self) -> None:
        runtime = _Runtime({})
        runtime.conversations["owner-a"] = (
            {"role": "user", "content": "Find Shanghai jobs"},
            {"role": "assistant", "content": "found: jobs were found"},
        )
        runtime.conversations["owner-b"] = (
            {"role": "user", "content": "Find Beijing jobs"},
        )
        app = FastAPI()
        app.include_router(build_job_agent_router(runtime))
        client = TestClient(app)
        owner_a = {"jobscope_job_agent_session": "owner-a-session"}
        owner_b = {"jobscope_job_agent_session": "owner-b-session"}

        before = client.get("/v1/job-agent/conversation", cookies=owner_a)
        cleared = client.delete("/v1/job-agent/conversation", cookies=owner_a)
        after = client.get("/v1/job-agent/conversation", cookies=owner_a)
        other_owner = client.get("/v1/job-agent/conversation", cookies=owner_b)

        self.assertEqual("Find Shanghai jobs", before.json()["turns"][0]["content"])
        self.assertEqual(204, cleared.status_code)
        self.assertEqual([], after.json()["turns"])
        self.assertEqual("Find Beijing jobs", other_owner.json()["turns"][0]["content"])


if __name__ == "__main__":
    unittest.main()
