"""Application/owner scoped documents with append-only revisions and CAS edits."""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from app.application.document_workspace import (
    ReportContent, ReportEdit, ReportConflictError, ReportNotFoundError,
    ReportUnavailableError, preview_edit,
)


class PostgresReportStore:
    def __init__(self, dsn: str, *, application: str):
        if application not in {"jobscope", "serviceflow"} or not dsn:
            raise ValueError("report store configuration invalid")
        self._dsn, self._application = dsn, application

    @contextmanager
    def connection(self):
        try:
            with psycopg.connect(self._dsn, connect_timeout=5) as connection:
                connection.execute("SET LOCAL statement_timeout='10s'")
                yield connection
        except psycopg.Error as error:
            raise ReportUnavailableError("report-storage-unavailable") from error

    def setup(self):
        with self.connection() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS agent_report_document (
                application text NOT NULL, document_id uuid NOT NULL, owner_id text NOT NULL,
                revision integer NOT NULL CHECK(revision>0), content jsonb NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY(application,document_id))""")
            c.execute("""CREATE TABLE IF NOT EXISTS agent_report_revision (
                application text NOT NULL, document_id uuid NOT NULL, revision integer NOT NULL,
                content jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY(application,document_id,revision),
                FOREIGN KEY(application,document_id) REFERENCES agent_report_document ON DELETE CASCADE)""")
            c.execute("""CREATE TABLE IF NOT EXISTS agent_report_proposal (
                application text NOT NULL, proposal_id uuid NOT NULL, document_id uuid NOT NULL,
                base_revision integer NOT NULL, content jsonb NOT NULL, expires_at timestamptz NOT NULL,
                confirmed_revision integer, PRIMARY KEY(application,proposal_id),
                FOREIGN KEY(application,document_id) REFERENCES agent_report_document ON DELETE CASCADE)""")
            c.execute("CREATE INDEX IF NOT EXISTS agent_report_owner_idx ON agent_report_document(application,owner_id,created_at DESC)")

    @staticmethod
    def view(row):
        return {"document_id": str(row[0]), "revision": row[1],
                "content": ReportContent.model_validate(row[2]).model_dump(mode="json")}

    def create(self, owner: str, content: ReportContent):
        if not owner.strip():
            raise ValueError("report owner required")
        key = uuid4()
        with self.connection() as c:
            c.execute("INSERT INTO agent_report_document(application,document_id,owner_id,revision,content) VALUES(%s,%s,%s,1,%s)",
                (self._application,key,owner,Jsonb(content.model_dump(mode="json"))))
            c.execute("INSERT INTO agent_report_revision(application,document_id,revision,content) VALUES(%s,%s,1,%s)",
                (self._application,key,Jsonb(content.model_dump(mode="json"))))
        return self.view((key,1,content.model_dump(mode="json")))

    def get(self, owner, key):
        with self.connection() as c:
            row = c.execute("SELECT document_id,revision,content FROM agent_report_document WHERE application=%s AND document_id=%s AND owner_id=%s",
                (self._application,key,owner)).fetchone()
            if row is None:
                raise ReportNotFoundError("report-not-found")
            return self.view(row)

    def list(self, owner):
        with self.connection() as c:
            rows = c.execute("SELECT document_id,revision,content FROM agent_report_document WHERE application=%s AND owner_id=%s ORDER BY created_at DESC,document_id DESC LIMIT 50",
                (self._application,owner)).fetchall()
        return [self.view(row) for row in rows]

    def propose(self, owner, key, edit: ReportEdit):
        current = self.get(owner,key)
        if current["revision"] != edit.base_revision:
            raise ReportConflictError("report-version-conflict")
        proposed = preview_edit(ReportContent.model_validate(current["content"]),edit)
        proposal = uuid4()
        with self.connection() as c:
            # Ownership/version must still hold after the preview was computed.
            row = c.execute("SELECT revision FROM agent_report_document WHERE application=%s AND document_id=%s AND owner_id=%s FOR SHARE",
                (self._application,key,owner)).fetchone()
            if row is None:
                raise ReportNotFoundError("report-not-found")
            if row[0] != edit.base_revision:
                raise ReportConflictError("report-version-conflict")
            c.execute("INSERT INTO agent_report_proposal(application,proposal_id,document_id,base_revision,content,expires_at) VALUES(%s,%s,%s,%s,%s,%s)",
                (self._application,proposal,key,edit.base_revision,Jsonb(proposed.model_dump(mode="json")),datetime.now(UTC)+timedelta(hours=1)))
        return {"proposal_id":str(proposal),"base_revision":edit.base_revision,
                "before":current["content"],"after":proposed.model_dump(mode="json")}

    def confirm(self, owner, key, proposal):
        with self.connection() as c:
            doc = c.execute("SELECT revision FROM agent_report_document WHERE application=%s AND document_id=%s AND owner_id=%s FOR UPDATE",
                (self._application,key,owner)).fetchone()
            if doc is None:
                raise ReportNotFoundError("report-not-found")
            row = c.execute("SELECT base_revision,content,expires_at,confirmed_revision FROM agent_report_proposal WHERE application=%s AND document_id=%s AND proposal_id=%s FOR UPDATE",
                (self._application,key,proposal)).fetchone()
            if row is None:
                raise ReportNotFoundError("report-proposal-not-found")
            base, content, expires, confirmed = row
            if confirmed is not None:
                return self.view((key,confirmed,content))
            if expires <= datetime.now(UTC):
                raise ReportConflictError("report-proposal-expired")
            if doc[0] != base:
                raise ReportConflictError("report-version-conflict")
            revision=base+1
            c.execute("UPDATE agent_report_document SET revision=%s,content=%s,updated_at=now() WHERE application=%s AND document_id=%s AND owner_id=%s AND revision=%s",
                (revision,Jsonb(content),self._application,key,owner,base))
            c.execute("INSERT INTO agent_report_revision(application,document_id,revision,content) VALUES(%s,%s,%s,%s)",
                (self._application,key,revision,Jsonb(content)))
            c.execute("UPDATE agent_report_proposal SET confirmed_revision=%s WHERE application=%s AND proposal_id=%s",
                (revision,self._application,proposal))
        return self.view((key,revision,content))

    def revisions(self, owner,key):
        self.get(owner,key)
        with self.connection() as c:
            rows=c.execute("SELECT r.document_id,r.revision,r.content FROM agent_report_revision r JOIN agent_report_document d USING(application,document_id) WHERE r.application=%s AND r.document_id=%s AND d.owner_id=%s ORDER BY r.revision DESC LIMIT 50",
                (self._application,key,owner)).fetchall()
        return [self.view(row) for row in rows]

    def delete(self, owner,key):
        with self.connection() as c:
            result=c.execute("DELETE FROM agent_report_document WHERE application=%s AND document_id=%s AND owner_id=%s",
                (self._application,key,owner))
            if result.rowcount != 1:
                raise ReportNotFoundError("report-not-found")
