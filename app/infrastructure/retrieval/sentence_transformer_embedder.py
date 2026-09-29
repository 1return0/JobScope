from __future__ import annotations

from pathlib import Path
from typing import Any

from app.application.retrieval.dense_retrieval import (
    EmbeddingSpec,
    EmbeddingVector,
)


class SentenceTransformerDependencyError(RuntimeError):
    pass


class SentenceTransformerTextEmbedder:
    def __init__(
        self,
        spec: EmbeddingSpec,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        cache_folder: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SentenceTransformerDependencyError(
                "install the 'embedding' optional dependencies"
            ) from error

        self._spec = spec
        self._batch_size = batch_size
        self._model: Any = SentenceTransformer(
            spec.model_name,
            revision=spec.model_revision,
            device=device,
            cache_folder=(
                None if cache_folder is None else str(cache_folder)
            ),
            local_files_only=local_files_only,
        )
        self._model.max_seq_length = spec.max_sequence_length
        get_dimension = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        )
        if get_dimension is None:
            get_dimension = (
                self._model.get_sentence_embedding_dimension
            )
        actual_dimension = get_dimension()
        if actual_dimension != spec.dimension:
            raise ValueError(
                "loaded model dimension does not match EmbeddingSpec"
            )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[EmbeddingVector, ...]:
        prepared = tuple(
            self._spec.document_instruction + text
            for text in texts
        )
        return self._encode_documents(prepared)

    def embed_query(self, query: str) -> EmbeddingVector:
        prepared = self._spec.query_instruction + query
        return self._encode_query(prepared)

    def _encode_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[EmbeddingVector, ...]:
        vectors = self._model.encode_document(
            list(texts),
            prompt="",
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return tuple(
            tuple(float(value) for value in vector)
            for vector in vectors
        )

    def _encode_query(self, query: str) -> EmbeddingVector:
        vector = self._model.encode_query(
            query,
            prompt="",
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return tuple(float(value) for value in vector)
