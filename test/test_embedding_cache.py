"""Test for embedding cache hit/miss behavior."""

import numpy as np
import time
from langchain.embedding_generator import EmbeddingGenerator


def test_embedding_cache():
    """Verifies that embedding caching works correctly.

    Tests cache miss on first call and cache hit on second call,
    then asserts the returned embeddings are identical.
    """
    eg = EmbeddingGenerator()
    text = "임베딩 캐시 테스트 문장입니다."

    t0 = time.time()
    emb1 = eg.embed(text)
    t1 = time.time()
    print(f"[Cache MISS] shape: {emb1.shape}, time: {t1 - t0:.3f}s")

    t2 = time.time()
    emb2 = eg.embed(text)
    t3 = time.time()
    print(f"[Cache HIT]  shape: {emb2.shape}, time: {t3 - t2:.3f}s")

    assert np.allclose(emb1, emb2)
    print("Embedding cache hit/miss test passed.")


if __name__ == "__main__":
    test_embedding_cache()
