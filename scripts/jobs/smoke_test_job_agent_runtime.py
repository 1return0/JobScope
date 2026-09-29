from __future__ import annotations

import argparse
import json
from typing import Sequence

from app.application.jobs.job_agent_planning import JobAgentPlan
from app.application.jobs.job_tool_graph import AgentToolCall
from app.config import Settings, load_settings
from app.runtime_composition import compose_job_agent_runtime


class _DeterministicRuntimeSmokePlanner:
    identity = "deterministic-job-agent-runtime-smoke-v1"

    def plan(self, *, user_query, tools) -> JobAgentPlan:
        return JobAgentPlan(
            decision="call_tool",
            reason="Verify the runtime database query chain.",
            tool_call=AgentToolCall(
                call_id="runtime-smoke-call-1",
                name="search_current_jobs",
                arguments={
                    "company": "__jobscope_runtime_smoke_no_match__",
                    "limit": 1,
                },
            ),
        )


def run_smoke(settings: Settings) -> dict[str, object]:
    runtime = compose_job_agent_runtime(
        settings,
        planner=_DeterministicRuntimeSmokePlanner(),
    )
    try:
        state = runtime.invoke("Verify current-job database connectivity")
    finally:
        runtime.close()
    return {
        "status": "passed",
        "planner_identity": state["planner_identity"],
        "planning_status": state["planning_status"],
        "tool_name": state["tool_result"]["tool_name"],
        "tool_output_status": state["tool_result"]["output"]["status"],
        "result_count": state["tool_result"]["output"]["result_count"],
        "tool_error": state["tool_error"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test Job Agent runtime against PostgreSQL."
    )
    parser.add_argument("--dotenv-override", action="store_true")
    arguments = parser.parse_args(argv)
    payload = run_smoke(
        load_settings(dotenv_override=arguments.dotenv_override)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
