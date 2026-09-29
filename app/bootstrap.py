from __future__ import annotations

from app.application.documents.document_parsing import (
    DocumentParserRegistry,
    DocumentParsingService,
)
from app.application.documents.corpus_ingestion import CorpusIngestionService
from app.application.documents.document_processing import (
    DocumentProcessingService,
)
from app.application.ocr.pdf_ocr_fallback import PdfOcrFallbackParser
from app.application.ocr.pdf_ocr_parsing import (
    OcrParsedDocumentAssembler,
    SingleColumnOcrReadingOrder,
)
from app.application.retrieval.current_corpus_search import (
    CurrentCorpusSearchService,
)
from app.application.retrieval.current_corpus_dense_search import (
    CurrentCorpusDenseSearchService,
)
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchService,
)
from app.application.retrieval.dense_retrieval import TextEmbedder
from app.application.retrieval.dense_retrieval_experiment import (
    CurrentCorpusDenseEvaluator,
)
from app.application.retrieval.lexical_retrieval import Bm25Config
from app.application.retrieval.hybrid_retrieval_experiment import (
    CurrentCorpusHybridEvaluator,
)
from app.application.retrieval.retrieval_experiment import (
    CurrentCorpusBm25Evaluator,
)
from app.config import Settings
from app.domain.documents.document_chunking import (
    ChunkingConfig,
    StructureAwareChunker,
)
from app.domain.sources.source_trust import (
    VerifiedDomainRegistry,
)
from app.infrastructure.persistence.database import (
    create_postgresql_engine,
    create_session_factory,
)
from app.infrastructure.documents.docx_document_parser import DocxDocumentParser
from app.infrastructure.persistence.document_corpus_repository import (
    SqlAlchemyDocumentCorpusRepository,
)
from app.infrastructure.documents.html_document_parser import HtmlDocumentParser
from app.infrastructure.documents.pdf_document_parser import PdfDocumentParser
from app.infrastructure.ocr.paddle_ocr_engine import PaddleOcrTextEngine
from app.infrastructure.ocr.paddle_ocr_pipeline_predictor import (
    build_paddle_ocr_pipeline_predictor,
)
from app.infrastructure.ocr.pdfium_page_renderer import (
    PdfiumPageRendererConfig,
    build_pdfium_page_renderer,
)
from app.infrastructure.documents.pptx_document_parser import PptxDocumentParser
from app.infrastructure.persistence.verified_domain_repository import (
    SqlAlchemyVerifiedDomainRepository,
)
from app.infrastructure.documents.xlsx_document_parser import XlsxDocumentParser


def build_document_parser_registry(
    settings: Settings | None = None,
) -> DocumentParserRegistry:
    pdf_parser = _build_pdf_parser(settings or Settings())
    return DocumentParserRegistry(
        [
            HtmlDocumentParser(),
            pdf_parser,
            DocxDocumentParser(),
            XlsxDocumentParser(),
            PptxDocumentParser(),
        ]
    )


def build_document_parsing_service(
    settings: Settings | None = None,
) -> DocumentParsingService:
    return DocumentParsingService(
        build_document_parser_registry(settings)
    )


def build_document_processing_service(
    settings: Settings | None = None,
    *,
    parsing_service: DocumentParsingService | None = None,
) -> DocumentProcessingService:
    selected_parsing_service = parsing_service
    if selected_parsing_service is None:
        selected_parsing_service = build_document_parsing_service(
            settings
        )
    return DocumentProcessingService(
        selected_parsing_service,
        StructureAwareChunker(),
    )


def build_corpus_ingestion_service(
    settings: Settings,
) -> CorpusIngestionService:
    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    return CorpusIngestionService(
        build_document_processing_service(settings),
        SqlAlchemyDocumentCorpusRepository(session_factory),
    )


def _build_pdf_parser(settings: Settings):
    primary_parser = PdfDocumentParser()
    if not settings.pdf_ocr_enabled:
        return primary_parser

    predictor = build_paddle_ocr_pipeline_predictor(
        engine=settings.pdf_ocr_engine,
        cache_folder=settings.pdf_ocr_cache_folder,
    )
    return PdfOcrFallbackParser(
        primary_parser,
        build_pdfium_page_renderer(
            PdfiumPageRendererConfig(
                dpi=settings.pdf_ocr_render_dpi,
                max_pages=settings.pdf_ocr_max_pages,
                max_pixels_per_page=(
                    settings.pdf_ocr_max_pixels_per_page
                ),
            )
        ),
        PaddleOcrTextEngine(
            predictor,
            minimum_confidence=(
                settings.pdf_ocr_minimum_confidence
            ),
        ),
        OcrParsedDocumentAssembler(
            SingleColumnOcrReadingOrder()
        ),
    )


def build_current_corpus_search_service(
    settings: Settings,
    *,
    chunker_version: str = ChunkingConfig().identity,
    bm25_config: Bm25Config = Bm25Config(),
) -> CurrentCorpusSearchService:
    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    repository = SqlAlchemyDocumentCorpusRepository(
        session_factory
    )
    return CurrentCorpusSearchService(
        repository,
        chunker_version=chunker_version,
        bm25_config=bm25_config,
    )


def build_current_corpus_dense_search_service(
    settings: Settings,
    embedder: TextEmbedder,
    *,
    chunker_version: str = ChunkingConfig().identity,
) -> CurrentCorpusDenseSearchService:
    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    repository = SqlAlchemyDocumentCorpusRepository(
        session_factory
    )
    return CurrentCorpusDenseSearchService(
        repository,
        embedder,
        chunker_version=chunker_version,
    )


def build_current_corpus_dense_evaluator(
    settings: Settings,
    embedder: TextEmbedder,
) -> CurrentCorpusDenseEvaluator:
    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    repository = SqlAlchemyDocumentCorpusRepository(
        session_factory
    )
    return CurrentCorpusDenseEvaluator(repository, embedder)


def build_current_corpus_hybrid_evaluator(
    settings: Settings,
    embedder: TextEmbedder,
    *,
    bm25_config: Bm25Config = Bm25Config(),
) -> CurrentCorpusHybridEvaluator:
    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    repository = SqlAlchemyDocumentCorpusRepository(
        session_factory
    )
    return CurrentCorpusHybridEvaluator(
        repository,
        embedder,
        bm25_config=bm25_config,
    )


def build_current_corpus_hybrid_search_service(
    settings: Settings,
    embedder: TextEmbedder,
    *,
    chunker_version: str = ChunkingConfig().identity,
    candidate_k: int = 20,
    rank_constant: int = 60,
    max_concurrent_queries: int = 1,
    bm25_config: Bm25Config = Bm25Config(),
) -> CurrentCorpusHybridSearchService:
    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    repository = SqlAlchemyDocumentCorpusRepository(
        session_factory
    )
    return CurrentCorpusHybridSearchService(
        repository,
        embedder,
        chunker_version=chunker_version,
        candidate_k=candidate_k,
        rank_constant=rank_constant,
        max_concurrent_queries=max_concurrent_queries,
        bm25_config=bm25_config,
    )


def build_current_corpus_bm25_evaluator(
    settings: Settings,
    *,
    bm25_config: Bm25Config = Bm25Config(),
) -> CurrentCorpusBm25Evaluator:
    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    repository = SqlAlchemyDocumentCorpusRepository(
        session_factory
    )
    return CurrentCorpusBm25Evaluator(
        repository,
        bm25_config=bm25_config,
    )


def build_verified_domain_registry(
    settings: Settings,
) -> VerifiedDomainRegistry:
    if settings.storage_backend == "csv":
        return VerifiedDomainRegistry.from_csv(
            settings.verified_domains_path
        )

    if settings.storage_backend == "postgresql":
        engine = create_postgresql_engine(settings)
        repository = SqlAlchemyVerifiedDomainRepository(
            create_session_factory(engine)
        )
        return VerifiedDomainRegistry(
            repository.list_current_records()
        )

    raise ValueError(
        "JOBSCOPE_STORAGE_BACKEND must be csv or postgresql"
    )
