import time
from embedding_generator import EmbeddingGenerator
from redis_handler import SemanticCacheHandler

def test_semantic_cache():
    embedding_gen = EmbeddingGenerator()
    cache = SemanticCacheHandler(
        embedding_model=embedding_gen.embeddings,
        redis_url="redis://host.docker.internal:6379",
        index_name="semantic_cache_test"
    )

    # 1. 질문-답변 쌍 저장
    question = "지구온난화의 주요 원인은 무엇인가요?"
    answer = "지구온난화의 주요 원인은 온실가스 배출, 특히 이산화탄소와 메탄입니다."
    meta = {"source": "테스트", "timestamp": time.time()}
    assert cache.save_qa_pair(question, answer, meta)

    # 2. 유사 질문으로 검색 (캐시 hit)
    similar_query = "지구 온난화가 왜 발생하나요?"
    hit_results = cache.search_similar_question(similar_query, score_threshold=0.7)
    print("[캐시 hit] 결과:", hit_results)
    assert any(r.get("answer") == answer and r.get("similarity", 0) > 0.2 for r in hit_results)

    # 3. 완전히 다른 질문 (캐시 miss)
    miss_query = "플라스틱 오염의 영향은?"
    miss_results = cache.search_similar_question(miss_query)
    print("[캐시 miss] 결과:", miss_results)

    highest_similarity_in_miss = max([r.get("similarity", 0) for r in miss_results]) if miss_results else 0
    assert not miss_results or highest_similarity_in_miss < 0.7

if __name__ == "__main__":
    test_semantic_cache() 