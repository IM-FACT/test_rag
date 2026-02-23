"""Test for semantic cache question-answer pair storage and retrieval."""

import time
from langchain.embedding_generator import EmbeddingGenerator
from langchain.redis_handler import SemanticCacheHandler


def test_semantic_cache():
    """Verifies semantic cache stores and retrieves Q&A pairs correctly.

    Tests:
        1. Saving a question-answer pair.
        2. Cache hit with a semantically similar question.
        3. Cache miss with an unrelated question.
    """
    embedding_gen = EmbeddingGenerator()
    cache = SemanticCacheHandler(
        embedding_model=embedding_gen.embeddings,
        redis_url="redis://host.docker.internal:6379",
        index_name="semantic_cache_test",
    )

    question = "지구온난화의 주요 원인은 무엇인가요?"
    answer = "지구온난화의 주요 원인은 온실가스 배출, 특히 이산화탄소와 메탄입니다."
    meta = {"source": "test", "timestamp": time.time()}
    assert cache.save_qa_pair(question, answer, meta)

    similar_query = "지구 온난화가 왜 발생하나요?"
    hit_results = cache.search_similar_question(similar_query, score_threshold=0.7)
    print(f"[Cache HIT] results: {hit_results}")
    assert any(
        r.get("answer") == answer and r.get("similarity", 0) > 0.2
        for r in hit_results
    )

    miss_query = "플라스틱 오염의 영향은?"
    miss_results = cache.search_similar_question(miss_query)
    print(f"[Cache MISS] results: {miss_results}")

    highest_similarity = (
        max(r.get("similarity", 0) for r in miss_results) if miss_results else 0
    )
    assert not miss_results or highest_similarity < 0.7


if __name__ == "__main__":
    test_semantic_cache()
