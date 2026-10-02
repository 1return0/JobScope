from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "JobScope"
    version: str = "0.1.0"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8110
    log_level: str = "INFO"
    max_upload_bytes: int = 10 * 1024 * 1024
    pdf_ocr_enabled: bool = False
    pdf_ocr_engine: str = "transformers"
    pdf_ocr_cache_folder: Path = (
        PROJECT_ROOT / ".model-cache" / "paddlex"
    )
    pdf_ocr_minimum_confidence: float = 0.5
    pdf_ocr_render_dpi: int = 300
    pdf_ocr_max_pages: int = 100
    pdf_ocr_max_pixels_per_page: int = 40_000_000
    verified_domains_path: Path = (
        PROJECT_ROOT / "data" / "verified-company-domains.csv"
    )
    storage_backend: str = "csv"
    database_host: str = "127.0.0.1"
    database_port: int = 5432
    database_name: str = "jobscope"
    database_user: str = "jobscope_app"
    database_password: str | None = field(default=None, repr=False)
    hybrid_retrieval_enabled: bool = False
    embedding_preset: str = "qwen3-embedding-0.6b"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 8
    embedding_cache_folder: Path = PROJECT_ROOT / ".model-cache"
    hybrid_candidate_k: int = 20
    hybrid_rank_constant: int = 60
    hybrid_max_concurrent_queries: int = 1
    answer_generation_enabled: bool = False
    answer_model_name: str = ""
    answer_model_base_url: str = ""
    answer_model_api_key_environment_variable: str = "DASHSCOPE_API_KEY"
    answer_model_timeout_seconds: float = 30.0
    answer_model_max_completion_tokens: int = 1200
    job_agent_enabled: bool = False
    job_agent_session_ttl_seconds: int = 7 * 24 * 60 * 60
    job_agent_session_cookie_secure: bool = False
    job_agent_harness_max_model_calls: int = 3
    job_agent_harness_max_tool_calls: int = 4
    job_agent_harness_max_elapsed_seconds: float = 30.0
    job_agent_planner_max_attempts: int = 2
    job_agent_planner_initial_backoff_seconds: float = 0.1
    cors_allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )


def load_settings(*, dotenv_override: bool = False) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=dotenv_override)
    cors_allowed_origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "JOBSCOPE_CORS_ALLOWED_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if origin.strip()
    )
    return Settings(
        environment=os.getenv("JOBSCOPE_ENV", "development"),
        host=os.getenv("JOBSCOPE_HOST", "127.0.0.1"),
        port=int(os.getenv("JOBSCOPE_PORT", "8110")),
        log_level=os.getenv("JOBSCOPE_LOG_LEVEL", "INFO").upper(),
        max_upload_bytes=int(
            os.getenv(
                "JOBSCOPE_MAX_UPLOAD_BYTES",
                str(10 * 1024 * 1024),
            )
        ),
        pdf_ocr_enabled=(
            os.getenv("JOBSCOPE_PDF_OCR_ENABLED", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        ),
        pdf_ocr_engine=os.getenv(
            "JOBSCOPE_PDF_OCR_ENGINE",
            "transformers",
        ).strip(),
        pdf_ocr_cache_folder=Path(
            os.getenv(
                "JOBSCOPE_PDF_OCR_CACHE_FOLDER",
                str(PROJECT_ROOT / ".model-cache" / "paddlex"),
            )
        ),
        pdf_ocr_minimum_confidence=float(
            os.getenv("JOBSCOPE_PDF_OCR_MINIMUM_CONFIDENCE", "0.5")
        ),
        pdf_ocr_render_dpi=int(
            os.getenv("JOBSCOPE_PDF_OCR_RENDER_DPI", "300")
        ),
        pdf_ocr_max_pages=int(
            os.getenv("JOBSCOPE_PDF_OCR_MAX_PAGES", "100")
        ),
        pdf_ocr_max_pixels_per_page=int(
            os.getenv(
                "JOBSCOPE_PDF_OCR_MAX_PIXELS_PER_PAGE",
                "40000000",
            )
        ),
        verified_domains_path=Path(
            os.getenv(
                "JOBSCOPE_VERIFIED_DOMAINS_PATH",
                str(
                    PROJECT_ROOT
                    / "data"
                    / "verified-company-domains.csv"
                ),
            )
        ),
        storage_backend=os.getenv(
            "JOBSCOPE_STORAGE_BACKEND",
            "csv",
        ).lower(),
        database_host=os.getenv(
            "JOBSCOPE_DB_HOST",
            "127.0.0.1",
        ),
        database_port=int(os.getenv("JOBSCOPE_DB_PORT", "5432")),
        database_name=os.getenv("JOBSCOPE_DB_NAME", "jobscope"),
        database_user=os.getenv("JOBSCOPE_DB_USER", "jobscope_app"),
        database_password=os.getenv("JOBSCOPE_DB_PASSWORD"),
        hybrid_retrieval_enabled=(
            os.getenv("JOBSCOPE_HYBRID_RETRIEVAL_ENABLED", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        ),
        embedding_preset=os.getenv(
            "JOBSCOPE_EMBEDDING_PRESET",
            "qwen3-embedding-0.6b",
        ),
        embedding_device=os.getenv(
            "JOBSCOPE_EMBEDDING_DEVICE",
            "cuda",
        ),
        embedding_batch_size=int(
            os.getenv("JOBSCOPE_EMBEDDING_BATCH_SIZE", "8")
        ),
        embedding_cache_folder=Path(
            os.getenv(
                "JOBSCOPE_EMBEDDING_CACHE_FOLDER",
                str(PROJECT_ROOT / ".model-cache"),
            )
        ),
        hybrid_candidate_k=int(
            os.getenv("JOBSCOPE_HYBRID_CANDIDATE_K", "20")
        ),
        hybrid_rank_constant=int(
            os.getenv("JOBSCOPE_HYBRID_RANK_CONSTANT", "60")
        ),
        hybrid_max_concurrent_queries=int(
            os.getenv("JOBSCOPE_HYBRID_MAX_CONCURRENT_QUERIES", "1")
        ),
        answer_generation_enabled=(
            os.getenv("JOBSCOPE_ANSWER_GENERATION_ENABLED", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        ),
        answer_model_name=os.getenv(
            "JOBSCOPE_ANSWER_MODEL_NAME",
            "",
        ).strip(),
        answer_model_base_url=os.getenv(
            "JOBSCOPE_ANSWER_MODEL_BASE_URL",
            "",
        ).strip(),
        answer_model_api_key_environment_variable=os.getenv(
            "JOBSCOPE_ANSWER_MODEL_API_KEY_ENV",
            "DASHSCOPE_API_KEY",
        ).strip(),
        answer_model_timeout_seconds=float(
            os.getenv("JOBSCOPE_ANSWER_MODEL_TIMEOUT_SECONDS", "30")
        ),
        answer_model_max_completion_tokens=int(
            os.getenv(
                "JOBSCOPE_ANSWER_MODEL_MAX_COMPLETION_TOKENS",
                "1200",
            )
        ),
        job_agent_enabled=(
            os.getenv("JOBSCOPE_JOB_AGENT_ENABLED", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        ),
        job_agent_session_ttl_seconds=int(
            os.getenv(
                "JOBSCOPE_JOB_AGENT_SESSION_TTL_SECONDS",
                str(7 * 24 * 60 * 60),
            )
        ),
        job_agent_session_cookie_secure=(
            os.getenv("JOBSCOPE_JOB_AGENT_SESSION_COOKIE_SECURE", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        ),
        job_agent_harness_max_model_calls=int(
            os.getenv("JOBSCOPE_JOB_AGENT_HARNESS_MAX_MODEL_CALLS", "3")
        ),
        job_agent_harness_max_tool_calls=int(
            os.getenv("JOBSCOPE_JOB_AGENT_HARNESS_MAX_TOOL_CALLS", "4")
        ),
        job_agent_harness_max_elapsed_seconds=float(
            os.getenv("JOBSCOPE_JOB_AGENT_HARNESS_MAX_ELAPSED_SECONDS", "30")
        ),
        job_agent_planner_max_attempts=int(
            os.getenv("JOBSCOPE_JOB_AGENT_PLANNER_MAX_ATTEMPTS", "2")
        ),
        job_agent_planner_initial_backoff_seconds=float(
            os.getenv(
                "JOBSCOPE_JOB_AGENT_PLANNER_INITIAL_BACKOFF_SECONDS", "0.1"
            )
        ),
        cors_allowed_origins=cors_allowed_origins,
    )
