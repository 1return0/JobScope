from __future__ import annotations

from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import Settings


class Base(DeclarativeBase):
    pass


def build_database_url(settings: Settings) -> URL:
    if not settings.database_password:
        raise ValueError(
            "JOBSCOPE_DB_PASSWORD is required for PostgreSQL storage"
        )

    return URL.create(
        drivername="postgresql+psycopg",
        username=settings.database_user,
        password=settings.database_password,
        host=settings.database_host,
        port=settings.database_port,
        database=settings.database_name,
    )


def build_postgresql_dsn(settings: Settings) -> str:
    """Build a Psycopg-compatible PostgreSQL DSN without exposing its password."""
    if not settings.database_password:
        raise ValueError(
            "JOBSCOPE_DB_PASSWORD is required for PostgreSQL storage"
        )

    dsn = URL.create(
        drivername="postgresql",
        username=settings.database_user,
        password=settings.database_password,
        host=settings.database_host,
        port=settings.database_port,
        database=settings.database_name,
    )
    return dsn.render_as_string(hide_password=False)


def create_postgresql_engine(settings: Settings) -> Engine:
    return create_engine(
        build_database_url(settings),
        pool_pre_ping=True,
    )


def create_session_factory(
    engine: Engine,
) -> sessionmaker:
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
