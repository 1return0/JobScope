from types import SimpleNamespace
from unittest.mock import patch
import subprocess

import pytest

from app.application.document_workspace import ReportConflictError,recruitment_report
from app.infrastructure.report_exporter import ReportExporter


def test_recruitment_reports_never_accept_unregistered_or_failed_tools():
    for state in ({"planning_status":"refused"},{"planning_status":"tool_selected",
        "tool_call":{"name":"delete_jobs"},"tool_result":{"output":{}}}):
        with pytest.raises(ReportConflictError):
            recruitment_report(state,"报告")


def test_docker_worker_has_no_network_mounts_credentials_or_shell():
    from tests.retrieval.test_report_workspace_live_smoke import content
    exporter=ReportExporter(backend="docker")
    with patch("app.infrastructure.report_exporter.subprocess.run") as run:
        run.return_value=SimpleNamespace(returncode=0,stdout=b"PKtest")
        exporter.export({"revision":1,"content":content().model_dump(mode="json")})
        command=run.call_args.args[0]
        assert "--network=none" in command and "--read-only" in command
        assert "--cap-drop=ALL" in command and "--user=65534:65534" in command
        assert "--memory=128m" in command and "--pids-limit=32" in command
        assert "-v" not in command and "--privileged" not in command
        assert run.call_args.kwargs["shell"] is False
        assert "DASHSCOPE_API_KEY" not in run.call_args.kwargs["env"]
