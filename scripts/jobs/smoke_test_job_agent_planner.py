from __future__ import annotations

import argparse
import json
from typing import Sequence

from app.application.jobs.job_agent_planning import (
    AgentToolDefinition,
    JobAgentPlanner,
)
from app.application.jobs.job_search_tool import (
    SearchCurrentJobsArguments,
    SearchCurrentJobsTool,
)
from app.config import load_settings
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    OpenAiCompatibleJobAgentPlannerConfig,
    build_openai_compatible_job_agent_planner,
)


def _search_tool_definition() -> AgentToolDefinition:
    return AgentToolDefinition(
        name=SearchCurrentJobsTool.name,
        description=SearchCurrentJobsTool.description,
        arguments_schema=SearchCurrentJobsArguments.model_json_schema(),
    )


def run_smoke(
    planner: JobAgentPlanner,
    *,
    user_query: str,
) -> dict:
    plan = planner.plan(
        user_query=user_query,
        tools=(_search_tool_definition(),),
    )
    return {
        "status": "passed",
        "planner_identity": planner.identity,
        "user_query": user_query,
        "plan": plan.model_dump(mode="json"),
        "limitations": [
            "single planner request",
            "no PostgreSQL query",
            "no tool execution",
            "no final grounded answer",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one real Job Agent planner decision smoke test."
    )
    parser.add_argument(
        "--query",
        default="帮我找上海的AI Agent实习岗位",
    )
    parser.add_argument(
        "--dotenv-override",
        action="store_true",
        help=(
            "Explicitly let the project .env replace inherited process "
            "variables for this smoke-test process only."
        ),
    )
    args = parser.parse_args(argv)

    settings = load_settings(dotenv_override=args.dotenv_override)
    planner = build_openai_compatible_job_agent_planner(
        OpenAiCompatibleJobAgentPlannerConfig(
            model_name=settings.answer_model_name,
            base_url=settings.answer_model_base_url,
            api_key_environment_variable=(
                settings.answer_model_api_key_environment_variable
            ),
            timeout_seconds=settings.answer_model_timeout_seconds,
            max_completion_tokens=min(
                settings.answer_model_max_completion_tokens,
                800,
            ),
            enable_thinking=False,
        )
    )
    print(
        json.dumps(
            run_smoke(planner, user_query=args.query),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
