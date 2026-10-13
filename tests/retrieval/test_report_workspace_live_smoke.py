import io
from types import SimpleNamespace
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from tests.retrieval.test_pgvector_live_smoke import vector_dsn
from app.api.report_workspace_api import build_report_router
from app.application.document_workspace import (
    ReportContent,ReportSection,ReportCitation,ReportEdit,ReportConflictError,ReportNotFoundError,
)
from app.infrastructure.postgres_report_store import PostgresReportStore
from app.infrastructure.report_exporter import ReportExporter


def content():
    return ReportContent(title="岗位对比报告",sections=(ReportSection(
        heading="Java 实习岗位",body="学历要求：本科及以上。",citations=(ReportCitation(
            source_reference="https://example.edu/recruitment",evidence_id="evidence-one",
            quoted_text="本科及以上"),)),))


def test_real_versions_ownership_confirmation_and_export(vector_dsn):
    store=PostgresReportStore(vector_dsn,application="jobscope")
    store.setup()
    app=FastAPI()
    app.include_router(build_report_router(prefix="/reports",application="jobscope",
        owner_resolver=lambda request:request.headers.get("Test-Owner","owner-a"),
        generator=lambda request,owner,payload:content(),settings=SimpleNamespace(),
        store=store,exporter=ReportExporter(backend="local")))
    with TestClient(app) as client:
        generated=client.post("/reports",json={"query":"比较岗位"})
        assert generated.status_code==201
        key=generated.json()["document_id"]
        assert client.get(f"/reports/{key}",headers={"Test-Owner":"owner-b"}).status_code==404
        proposal=client.post(f"/reports/{key}/proposals",json={
            "base_revision":1,"title":"修改后的报告","user_notes":"我更关注实习岗位。"}).json()
        assert client.get(f"/reports/{key}").json()["revision"]==1
        path=f"/reports/{key}/proposals/{proposal['proposal_id']}/confirm"
        assert client.post(path,json={"user_confirmed":False}).status_code==422
        confirmed=client.post(path,json={"user_confirmed":True})
        assert confirmed.status_code==200 and confirmed.json()["revision"]==2
        assert client.post(path,json={"user_confirmed":True}).json()["revision"]==2
        assert len(client.get(f"/reports/{key}/revisions").json()["items"])==2
        assert client.post(f"/reports/{key}/export",json={"expected_revision":1,"user_confirmed":True}).status_code==409
        exported=client.post(f"/reports/{key}/export",json={"expected_revision":2,"user_confirmed":True})
        assert exported.status_code==200
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            xml=archive.read("word/document.xml").decode()
            assert "修改后的报告" in xml and "https://example.edu/recruitment" in xml
            assert "用户批注" in xml and "本科及以上" in xml
            assert not any("vbaProject" in name for name in archive.namelist())
        assert client.delete(f"/reports/{key}").status_code==204
        assert client.get(f"/reports/{key}").status_code==404


def test_conflicting_proposals_and_application_isolation(vector_dsn):
    store=PostgresReportStore(vector_dsn,application="jobscope")
    store.setup()
    doc=store.create("owner-a",content())
    key=doc["document_id"]
    a=store.propose("owner-a",key,ReportEdit(base_revision=1,title="A"))
    b=store.propose("owner-a",key,ReportEdit(base_revision=1,title="B"))
    store.confirm("owner-a",key,a["proposal_id"])
    with pytest.raises(ReportConflictError):
        store.confirm("owner-a",key,b["proposal_id"])
    other=PostgresReportStore(vector_dsn,application="serviceflow")
    with pytest.raises(ReportNotFoundError):
        other.get("owner-a",key)
    with store.connection() as c:
        assert c.execute("SELECT count(*) FROM agent_report_revision").fetchone()[0]==2


def test_forbidden_credentials_and_fact_rewrites(vector_dsn):
    store=PostgresReportStore(vector_dsn,application="jobscope")
    store.setup()
    doc=store.create("owner-a",content())
    from pydantic import ValidationError
    from app.application.document_workspace import preview_edit
    with pytest.raises(ValidationError):
        ReportEdit(base_revision=1,body="伪造学历要求")
    with pytest.raises(ValidationError):
        preview_edit(content(),ReportEdit(base_revision=1,user_notes="Bearer temporary-secret"))
    with pytest.raises(ValueError):
        preview_edit(content(),ReportEdit(base_revision=1,section_order=("foreign-section",)))
    assert store.get("owner-a",doc["document_id"])["revision"]==1


def test_job_router_uses_server_cookie_owner_and_server_tool_result(vector_dsn):
    from app.api.jobs.job_report_api import build_job_report_router
    from app.application.jobs.job_agent_sessions import JobAgentSessionNotFoundError
    from app.config import Settings
    class Runtime:
        calls=0
        def resolve_browser_owner(self,token):
            if token!="test-cookie":
                raise JobAgentSessionNotFoundError("missing session")
            return "server-owner"
        def invoke(self,query,*,owner_id):
            assert owner_id=="server-owner"
            self.calls+=1
            return {"planning_status":"tool_selected","tool_call":{"name":"search_current_jobs"},
                    "tool_result":{"output":{"jobs":[{"record_id":"current-record",
                        "job_title":"Java Intern","company":"Example","locations":["Shanghai"],
                        "source_reference":"https://example.edu/jobs"}]}}}
    runtime=Runtime()
    app=FastAPI()
    app.state.job_agent_runtime=runtime
    app.include_router(build_job_report_router(Settings(document_reports_enabled=True,
        document_reports_dsn=vector_dsn,document_export_backend="local")))
    with TestClient(app) as client:
        assert client.post("/v1/job-agent/reports",json={"query":"find jobs"}).status_code==401
        assert runtime.calls==0
        client.cookies.set("jobscope_job_agent_session","test-cookie")
        assert client.post("/v1/job-agent/reports",json={"query":"find jobs","owner_id":"forged"}).status_code==422
        result=client.post("/v1/job-agent/reports",json={"query":"find jobs"})
        assert result.status_code==201 and runtime.calls==1
        assert result.json()["content"]["sections"][0]["heading"]=="Java Intern"
