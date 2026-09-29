from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from dotenv import load_dotenv

from app.config import PROJECT_ROOT


DEFAULT_QUERY = "帮我查找上海的AI Agent实习岗位"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the real Job Agent FastAPI chain."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--raise-server-exceptions",
        action="store_true",
        help="Expose the original server exception for diagnosis.",
    )
    arguments = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    os.environ["JOBSCOPE_JOB_AGENT_ENABLED"] = "true"
    os.environ["JOBSCOPE_HYBRID_RETRIEVAL_ENABLED"] = "false"
    os.environ["JOBSCOPE_ANSWER_GENERATION_ENABLED"] = "false"

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(
        app,
        raise_server_exceptions=arguments.raise_server_exceptions,
    ) as client:
        response = client.post(
            "/v1/job-agent/query",
            json={"user_query": arguments.query},
        )

    try:
        response_body = response.json()
    except ValueError:
        response_body = {
            "non_json_body": response.text[:500],
        }
    payload = {
        "http_status": response.status_code,
        "response": response_body,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
