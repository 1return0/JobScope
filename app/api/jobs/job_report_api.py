from app.api.report_workspace_api import build_report_router
from app.api.jobs.job_agent_api import JOB_AGENT_SESSION_COOKIE_NAME
from app.application.document_workspace import recruitment_report, ReportUnavailableError
from app.application.jobs.job_agent_sessions import JobAgentSessionNotFoundError


def build_job_report_router(settings):
    def runtime(request):
        value=getattr(request.app.state,"job_agent_runtime",None)
        if value is None:
            raise ReportUnavailableError("job-agent-unavailable")
        return value

    def owner(request):
        try:
            return runtime(request).resolve_browser_owner(request.cookies.get(JOB_AGENT_SESSION_COOKIE_NAME))
        except JobAgentSessionNotFoundError as error:
            raise PermissionError("report-authentication-required") from error

    def generate(request,owner_id,payload):
        state=runtime(request).invoke(payload.query,owner_id=owner_id)
        return recruitment_report(state,payload.title)

    return build_report_router(prefix="/v1/job-agent/reports",application="jobscope",
        owner_resolver=owner,generator=generate,settings=settings)
