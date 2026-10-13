"""Selected real session/model/pgvector/report/SSH-Docker chain; clean own data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--query",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from app.config import load_settings
    load_settings(dotenv_override=True)
    from app.main import app
    from fastapi.testclient import TestClient
    from sqlalchemy import delete
    from app.infrastructure.persistence.models import JobAgentSessionRow
    checks=[]
    document_id=None
    with TestClient(app) as client:
        runtime=app.state.job_agent_runtime
        response=client.post("/v1/job-agent/sessions")
        assert response.status_code==201,"real-session-issuance-failed"
        token=client.cookies.get("jobscope_job_agent_session")
        owner=runtime.resolve_browser_owner(token)
        checks.append("real-session")
        try:
            result=client.post("/v1/job-agent/reports",json={"query":args.query,"title":"真实链路验收报告"})
            assert result.status_code==201,f"report-generation-failed:{result.status_code}:{result.json().get('detail')}"
            document=result.json()
            document_id=document["document_id"]
            assert all(section["citations"] for section in document["content"]["sections"])
            checks.append("real-agent-pgvector-report-citations")
            proposal=client.post(f"/v1/job-agent/reports/{document_id}/proposals",json={"base_revision":1,"title":"验收确认版"})
            assert proposal.status_code==201
            confirmed=client.post(f"/v1/job-agent/reports/{document_id}/proposals/{proposal.json()['proposal_id']}/confirm",json={"user_confirmed":True})
            assert confirmed.status_code==200 and confirmed.json()["revision"]==2
            checks.append("postgres-version-confirmation")
            exported=client.post(f"/v1/job-agent/reports/{document_id}/export",json={"expected_revision":2,"user_confirmed":True})
            assert exported.status_code==200,f"docker-export-failed:{exported.status_code}"
            with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
                assert "验收确认版" in archive.read("word/document.xml").decode()
            checks.append("ssh-docker-docx")
            backend=client.get("/v1/job-agent/reports/capabilities").json()["export_backend"]
            assert backend=="ssh_docker"
            report={"status":"passed","checks":checks,"export_backend":backend,
                "sections":len(document["content"]["sections"]),"docx_bytes":len(exported.content),
                "boundary":"Real application in-process HTTP, real cookie identity/model/current corpus/pgvector/PG reports/SSH Docker. Not browser/load/production SLA."}
        finally:
            if document_id:
                assert client.delete(f"/v1/job-agent/reports/{document_id}").status_code==204
            runtime.clear_conversation_turns(owner_id=owner)
            with runtime.engine.begin() as connection:
                connection.execute(delete(JobAgentSessionRow).where(
                    JobAgentSessionRow.token_hash==hashlib.sha256(token.encode()).hexdigest(),
                    JobAgentSessionRow.owner_id==owner))
    report["cleanup"]="own report, revisions, proposals, conversation and issued session removed"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
