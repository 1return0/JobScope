from __future__ import annotations

from datetime import timedelta
import hashlib
import os

import pytest
from sqlalchemy import delete

from app.application.jobs.job_agent_sessions import JobAgentSessionService
from app.config import load_settings
from app.infrastructure.persistence.database import (
    create_postgresql_engine,
    create_session_factory,
)
from app.infrastructure.persistence.job_agent_session_repository import (
    SqlAlchemyJobAgentSessionStore,
)
from app.infrastructure.persistence.models import JobAgentSessionRow


pytestmark = pytest.mark.skipif(
    os.getenv("JOBSCOPE_RUN_POSTGRES_SESSION_SMOKE") != "true",
    reason="requires an explicitly enabled JobScope PostgreSQL smoke run",
)


def test_postgresql_session_store_never_persists_raw_browser_token() -> None:
    engine = create_postgresql_engine(load_settings())
    session_factory = create_session_factory(engine)
    service = JobAgentSessionService(
        SqlAlchemyJobAgentSessionStore(session_factory),
        ttl=timedelta(minutes=5),
    )
    issued = service.issue_session()
    token_hash = hashlib.sha256(
        issued.session_token.encode("utf-8")
    ).hexdigest()
    try:
        owner_id = service.resolve_owner_id(issued.session_token)
        with session_factory() as session:
            row = session.get(JobAgentSessionRow, token_hash)
            assert row is not None
            assert row.owner_id == owner_id
            assert row.token_hash != issued.session_token
    finally:
        with session_factory.begin() as session:
            session.execute(
                delete(JobAgentSessionRow).where(
                    JobAgentSessionRow.token_hash == token_hash
                )
            )
        engine.dispose()
