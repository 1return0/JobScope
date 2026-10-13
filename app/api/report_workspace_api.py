"""Shared HTTP lifecycle; each project supplies its own trusted identity/generator."""
from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import Field
from typing import Literal

from app.application.document_workspace import (
    StrictModel, ReportEdit, ReportConflictError, ReportNotFoundError, ReportUnavailableError,
)
from app.infrastructure.postgres_report_store import PostgresReportStore
from app.infrastructure.report_exporter import ReportExporter


class GenerateReportRequest(StrictModel):
    query: str = Field(min_length=1,max_length=1000)
    title: str = Field(default="来源核查报告",min_length=1,max_length=120)


class ConfirmReportRequest(StrictModel):
    user_confirmed: Literal[True]


class ExportReportRequest(ConfirmReportRequest):
    expected_revision: int = Field(ge=1)


def build_report_router(*,prefix,application,owner_resolver,generator,settings,store=None,exporter=None):
    router=APIRouter(prefix=prefix,tags=["document-reports"])

    @lru_cache(maxsize=1)
    def require_store():
        if store is not None:
            return store
        if not settings.document_reports_enabled or not settings.document_reports_dsn:
            raise ReportUnavailableError("document-reports-disabled")
        result=PostgresReportStore(settings.document_reports_dsn,application=application)
        result.setup()
        return result

    def execute(request,operation):
        try:
            owner=owner_resolver(request)
            return operation(owner,require_store())
        except ReportNotFoundError as error:
            raise HTTPException(404,str(error)) from error
        except ReportConflictError as error:
            raise HTTPException(409,str(error)) from error
        except ReportUnavailableError as error:
            raise HTTPException(503,str(error)) from error
        except PermissionError as error:
            raise HTTPException(401,"report-authentication-required") from error
        except ValueError as error:
            raise HTTPException(422,"invalid-report-request") from error
        except RuntimeError as error:
            raise HTTPException(503,"report-dependency-unavailable") from error

    @router.post("",status_code=201)
    def create(payload:GenerateReportRequest,request:Request):
        return execute(request,lambda owner,db:db.create(owner,generator(request,owner,payload)))

    @router.get("")
    def list_reports(request:Request):
        return execute(request,lambda owner,db:{"items":db.list(owner)})

    @router.get("/capabilities")
    def capabilities(request:Request):
        return execute(request,lambda owner,db:{"export_backend":
            exporter.backend if exporter is not None else settings.document_export_backend})

    @router.get("/{document_id}")
    def get(document_id:UUID,request:Request):
        return execute(request,lambda owner,db:db.get(owner,document_id))

    @router.post("/{document_id}/proposals",status_code=201)
    def propose(document_id:UUID,payload:ReportEdit,request:Request):
        return execute(request,lambda owner,db:db.propose(owner,document_id,payload))

    @router.post("/{document_id}/proposals/{proposal_id}/confirm")
    def confirm(document_id:UUID,proposal_id:UUID,payload:ConfirmReportRequest,request:Request):
        return execute(request,lambda owner,db:db.confirm(owner,document_id,proposal_id))

    @router.get("/{document_id}/revisions")
    def revisions(document_id:UUID,request:Request):
        return execute(request,lambda owner,db:{"items":db.revisions(owner,document_id)})

    @router.delete("/{document_id}",status_code=204)
    def delete(document_id:UUID,request:Request):
        execute(request,lambda owner,db:db.delete(owner,document_id))
        return Response(status_code=204)

    @router.post("/{document_id}/export")
    def export(document_id:UUID,payload:ExportReportRequest,request:Request):
        def operation(owner,db):
            document=db.get(owner,document_id)
            if document["revision"]!=payload.expected_revision:
                raise ReportConflictError("report-version-conflict")
            worker=exporter or ReportExporter(backend=settings.document_export_backend,
                ssh_config_file=settings.document_export_ssh_config_file)
            data=worker.export(document)
            return Response(data,media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition":f'attachment; filename="report-{document_id}-v{document["revision"]}.docx"',
                         "X-Report-Revision":str(document["revision"]),"Cache-Control":"no-store"})
        return execute(request,operation)
    return router
