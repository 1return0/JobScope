from app.application.retrieval.dense_retrieval import EmbeddingSpec


QWEN3_EMBEDDING_06B = EmbeddingSpec(
    provider="huggingface",
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_revision=(
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    ),
    dimension=1024,
    max_sequence_length=1024,
    normalization="l2",
    query_instruction=(
        "Instruct: Given a campus recruitment question, retrieve "
        "official evidence that answers the question.\nQuery: "
    ),
)


BGE_SMALL_ZH_V15 = EmbeddingSpec(
    provider="huggingface",
    model_name="BAAI/bge-small-zh-v1.5",
    model_revision=(
        "7999e1d3359715c523056ef9478215996d62a620"
    ),
    dimension=512,
    max_sequence_length=512,
    normalization="l2",
    query_instruction=(
        "为这个句子生成表示以用于检索相关文章："
    ),
)


EMBEDDING_MODEL_PRESETS = {
    "qwen3-embedding-0.6b": QWEN3_EMBEDDING_06B,
    "bge-small-zh-v1.5": BGE_SMALL_ZH_V15,
}
