import numpy as np
import time
from embedding_generator import EmbeddingGenerator

def test_embedding_cache():
    eg = EmbeddingGenerator()
    text = "임베딩 캐시 테스트 문장입니다."

    # 1. 캐시 miss (최초)
    t0 = time.time()
    emb1 = eg.get_embedding(text)
    t1 = time.time()
    print(f"[캐시 miss] 임베딩 shape: {emb1.shape}, 소요: {t1-t0:.3f}s")

    # 2. 캐시 hit (두번째)
    t2 = time.time()
    emb2 = eg.get_embedding(text)
    t3 = time.time()
    print(f"[캐시 hit] 임베딩 shape: {emb2.shape}, 소요: {t3-t2:.3f}s")

    # 3. 값 일치 확인
    assert np.allclose(emb1, emb2)
    print("임베딩 캐시 hit/miss 정상 동작!")

if __name__ == "__main__":
    test_embedding_cache() 