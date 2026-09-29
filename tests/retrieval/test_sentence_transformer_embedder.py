import sys
import types
import unittest
from unittest.mock import patch

from app.application.retrieval.dense_retrieval import EmbeddingSpec
from app.infrastructure.retrieval.embedding_model_presets import (
    BGE_SMALL_ZH_V15,
    QWEN3_EMBEDDING_06B,
)
from app.infrastructure.retrieval.sentence_transformer_embedder import (
    SentenceTransformerTextEmbedder,
)


class _FakeSentenceTransformer:
    instances = []

    def __init__(self, model_name, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.max_seq_length = None
        self.encoded_texts = []
        self.__class__.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def get_embedding_dimension(self) -> int:
        return 2

    def encode_document(self, texts, **kwargs):
        self.encoded_texts.append(("document", texts, kwargs))
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text, **kwargs):
        self.encoded_texts.append(("query", text, kwargs))
        return [1.0, 0.0]


class SentenceTransformerTextEmbedderTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSentenceTransformer.instances.clear()
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = _FakeSentenceTransformer
        self.module_patch = patch.dict(
            sys.modules,
            {"sentence_transformers": fake_module},
        )
        self.module_patch.start()

    def tearDown(self) -> None:
        self.module_patch.stop()

    def test_pins_revision_and_applies_explicit_instructions(self) -> None:
        spec = EmbeddingSpec(
            provider="test",
            model_name="test/model",
            model_revision="a" * 40,
            dimension=2,
            max_sequence_length=128,
            query_instruction="query: ",
            document_instruction="passage: ",
        )
        embedder = SentenceTransformerTextEmbedder(
            spec,
            device="cuda",
            local_files_only=True,
        )

        documents = embedder.embed_documents(("document",))
        query = embedder.embed_query("question")

        model = _FakeSentenceTransformer.instances[0]
        self.assertEqual("a" * 40, model.kwargs["revision"])
        self.assertTrue(model.kwargs["local_files_only"])
        self.assertEqual(128, model.max_seq_length)
        self.assertEqual(
            ["passage: document"],
            model.encoded_texts[0][1],
        )
        self.assertEqual(
            "query: question",
            model.encoded_texts[1][1],
        )
        self.assertEqual("", model.encoded_texts[0][2]["prompt"])
        self.assertEqual("", model.encoded_texts[1][2]["prompt"])
        self.assertEqual(((1.0, 0.0),), documents)
        self.assertEqual((1.0, 0.0), query)

    def test_rejects_loaded_model_with_wrong_dimension(self) -> None:
        spec = EmbeddingSpec(
            provider="test",
            model_name="test/model",
            model_revision="a" * 40,
            dimension=3,
        )

        with self.assertRaisesRegex(ValueError, "dimension"):
            SentenceTransformerTextEmbedder(spec)

    def test_presets_pin_full_commits_and_expected_dimensions(self) -> None:
        self.assertEqual(40, len(QWEN3_EMBEDDING_06B.model_revision))
        self.assertEqual(1024, QWEN3_EMBEDDING_06B.dimension)
        self.assertEqual(40, len(BGE_SMALL_ZH_V15.model_revision))
        self.assertEqual(512, BGE_SMALL_ZH_V15.dimension)
        self.assertNotEqual(
            QWEN3_EMBEDDING_06B.identity,
            BGE_SMALL_ZH_V15.identity,
        )


if __name__ == "__main__":
    unittest.main()
