"""Browser + real HTTP/PostgreSQL report smoke, with explicit Stub identity/generation.

No production Corpus, JWT, Checkpoint or model is used. A fresh schema is removed.
Requires Playwright and an installed Edge browser; never opens a personal profile.
"""
import argparse
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ["PYTHON_DOTENV_DISABLED"]="true"

from dotenv import dotenv_values
from fastapi import FastAPI,Request,Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import uvicorn

from app.api.report_workspace_api import build_report_router
from app.application.document_workspace import ReportContent,ReportSection,ReportCitation
from app.infrastructure.postgres_report_store import PostgresReportStore
from app.infrastructure.report_exporter import ReportExporter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--connection-file",type=Path,required=True)
    parser.add_argument("--application",choices=("jobscope","serviceflow"),default="jobscope")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    dsn=dotenv_values(args.connection_file)["AGENT_PGVECTOR_SMOKE_DSN"]
    schema="agent_report_browser_"+uuid.uuid4().hex
    assert re.fullmatch(r"agent_report_browser_[0-9a-f]{32}",schema)
    with psycopg.connect(dsn,autocommit=True) as c:
        c.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    server=None
    try:
        store=PostgresReportStore(make_conninfo(dsn,options=f"-c search_path={schema},public"),application=args.application)
        store.setup()
        dist=(ROOT/"frontend/dist") if args.application=="jobscope" else ROOT.parent/"frontend/dist"
        app=FastAPI()
        sessions={}
        def owner(request):
            if args.application=="serviceflow":
                if request.headers.get("Authorization")!="Bearer browser-probe":
                    raise PermissionError("stub authentication required")
                return "browser-owner"
            token=request.cookies.get("jobscope_job_agent_session")
            if token not in sessions:
                raise PermissionError("stub authentication required")
            return sessions[token]
        def generate(request,owner_id,payload):
            heading="Java 实习岗位" if args.application=="jobscope" else "校园网络访问指南"
            body="学历要求：本科及以上。" if args.application=="jobscope" else "依据公开指南办理校园网络访问。"
            return ReportContent(title=payload.title,sections=(ReportSection(heading=heading,
                body=body,citations=(ReportCitation(
                    source_reference="https://example.edu/recruitment",evidence_id="browser-probe",
                    quoted_text="本科及以上"),)),))
        from types import SimpleNamespace
        prefix="/v1/job-agent/reports" if args.application=="jobscope" else "/api/v1/knowledge/reports"
        app.include_router(build_report_router(prefix=prefix,application=args.application,
            owner_resolver=owner,generator=generate,settings=SimpleNamespace(),store=store,
            exporter=ReportExporter(backend="local")))
        @app.post("/v1/job-agent/sessions")
        def issue(response:Response):
            token=uuid.uuid4().hex
            sessions[token]=uuid.uuid4().hex
            response.set_cookie("jobscope_job_agent_session",token,httponly=True,samesite="lax")
            return {"expires_at":"2099-01-01T00:00:00Z"}
        @app.get("/reports")
        def page():
            return FileResponse(dist/"index.html")
        app.mount("/",StaticFiles(directory=dist,html=True))
        port=8001 if args.application=="serviceflow" else 18110
        server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="error",access_log=False))
        # Never take over or stop a port owned by an existing user service.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1",port))==0:
                raise RuntimeError("browser smoke port already in use")
        thread=threading.Thread(target=server.run,daemon=True)
        thread.start()
        for _ in range(50):
            if server.started: break
            time.sleep(.1)
        assert server.started
        with sync_playwright() as p:
            browser=p.chromium.launch(channel="msedge",headless=True)
            context=browser.new_context(viewport={"width":1360,"height":1100},accept_downloads=True)
            if args.application=="serviceflow":
                context.add_init_script("sessionStorage.setItem('serviceflow.access-token','browser-probe')")
            page=context.new_page()
            errors=[]
            page.on("pageerror",lambda error:errors.append(str(error)))
            page.on("dialog",lambda dialog:dialog.accept())
            page.goto(f"http://127.0.0.1:{port}/reports")
            page.get_by_text("暂无报告",exact=True).wait_for()
            page.get_by_label("报告标题",exact=True).fill("浏览器证据报告")
            page.get_by_label("查询与汇总需求",exact=True).fill("Compare internship evidence")
            page.get_by_role("button",name="生成证据草稿",exact=True).click()
            page.get_by_role("heading",name="浏览器证据报告 · v1",exact=True).wait_for()
            page.get_by_label("报告标题",exact=True).fill("浏览器修改报告")
            page.get_by_label("个人批注（不视为已验证事实）",exact=True).fill("只用于验收的个人批注。")
            page.get_by_role("button",name="预览修改",exact=True).click()
            page.get_by_role("button",name="确认保存新版本",exact=True).click()
            page.get_by_role("heading",name="浏览器修改报告 · v2",exact=True).wait_for()
            page.reload()
            page.get_by_role("button",name="浏览器修改报告 · v2",exact=True).click()
            page.get_by_role("heading",name="浏览器修改报告 · v2",exact=True).wait_for()
            with page.expect_download() as info:
                page.get_by_role("button",name="确认并导出 Word",exact=True).click()
            download=info.value
            data=Path(download.path()).read_bytes()
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                assert "浏览器修改报告" in archive.read("word/document.xml").decode()
            args.output.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.output/"report-workspace.png"),full_page=True)
            page.get_by_role("button",name="删除报告",exact=True).click()
            page.get_by_text("暂无报告",exact=True).wait_for()
            assert not errors
            browser.close()
        result={"status":"passed","application":args.application,
            "checks":["generate","preview","confirm-version-2","reload-persistence","download-docx","delete"],
            "page_errors":0,"boundary":"Real Vue/HTTP/PostgreSQL/export; Stub identity and generator, NOT real JWT/model or Docker isolation"}
        (args.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
        print(json.dumps(result,indent=2))
    finally:
        if server is not None:
            server.should_exit=True
            if 'thread' in locals(): thread.join(timeout=10)
        with psycopg.connect(dsn,autocommit=True) as c:
            c.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


if __name__=="__main__":
    main()
