"""Bounded fixed worker; local mode is NOT an OS sandbox. Docker mode is opt-in."""
import json
import os
from pathlib import Path
import subprocess
import sys
import re
from uuid import uuid4

from app.application.document_workspace import ReportContent, ReportUnavailableError
from app.infrastructure.report_export_policy import docker_worker_command


class ReportExporter:
    def __init__(self, *, backend="disabled", image="agent-report-export:local", ssh_config_file=None):
        if backend not in {"disabled","local","docker","ssh_docker"}:
            raise ValueError("unsupported report export backend")
        if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*",image):
            raise ValueError("invalid report worker image")
        self.backend,self.image=backend,image
        self.ssh_config_file=ssh_config_file

    def export(self, report):
        ReportContent.model_validate(report["content"])
        raw=json.dumps(report,ensure_ascii=False).encode()
        if len(raw)>100_000:
            raise ValueError("report export input too large")
        if self.backend=="disabled":
            raise ReportUnavailableError("report-export-disabled")
        if self.backend=="ssh_docker":
            if not self.ssh_config_file:
                raise ReportUnavailableError("report-export-ssh-config-required")
            from app.infrastructure.ssh_report_exporter import export_over_ssh
            return export_over_ssh(raw,config_file=self.ssh_config_file,image=self.image)
        name="agent-report-export-"+uuid4().hex
        if self.backend=="docker":
            command=docker_worker_command(name,self.image)
        else:
            # Developer mode allows installed user-site dependencies. It is
            # deliberately not called a sandbox; production Docker uses -I.
            command=[sys.executable,"-E",str(Path(__file__).with_name("report_export_worker.py"))]
        # Never inherit API keys, JWTs or database credentials into the worker.
        environment={key:os.environ[key] for key in
            ("PATH","SYSTEMROOT","TEMP","TMP","APPDATA","LOCALAPPDATA","USERPROFILE")
            if key in os.environ}
        try:
            result=subprocess.run(command,input=raw,capture_output=True,timeout=15,
                                  check=False,env=environment,shell=False)
        except subprocess.TimeoutExpired as error:
            if self.backend=="docker":
                # Unique container created for this invocation only; no volumes.
                try:
                    subprocess.run(["docker","rm","-f",name],capture_output=True,
                                   timeout=5,check=False,env=environment,shell=False)
                except (OSError,subprocess.TimeoutExpired):
                    pass
            raise ReportUnavailableError("report-export-timeout") from error
        except OSError as error:
            raise ReportUnavailableError("report-export-worker-unavailable") from error
        if result.returncode or len(result.stdout)>2_000_000 or not result.stdout.startswith(b"PK"):
            raise ReportUnavailableError("report-export-worker-failed")
        return result.stdout
