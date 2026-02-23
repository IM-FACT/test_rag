"""Test for Redis 8 Vector Search functionality."""

from langchain.embedding_generator import EmbeddingGenerator
from langchain.redis_handler import RedisVectorSearchHandler
import uuid


def test_vector_search():
    """Tests the full vector search pipeline.

    Steps:
        1. Initialize embedding generator and Redis handler.
        2. Store test documents with embeddings.
        3. Run similarity searches and verify results.
        4. Display index metadata.
    """
    print("\n=== Redis 8 Vector Search Test ===\n")

    # Step 1: Initialize components
    print("[1/5] Initializing components...")
    try:
        embedding_gen = EmbeddingGenerator()
        redis_handler = RedisVectorSearchHandler(
            embedding_model=embedding_gen.embeddings,
            redis_url="redis://localhost:6379",
            index_name="document_index",
        )
        print("Components initialized.\n")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Step 2: Prepare test data
    print("[2/5] Preparing test data...")
    test_documents = [
        {
            "text": "종이 빨대는 플라스틱 빨대의 대안으로 사용되지만, 제조 과정에서 화학물질을 사용하고 재활용이 어려운 단점이 있습니다.",
            "metadata": {
                "question": "종이 빨대가 정말 친환경적인가요?",
                "source_url": "https://example.com/paper-straws",
                "category": "플라스틱 대체품",
            },
        },
        {
            "text": "태양광 패널은 일반적으로 25-30년의 수명을 가지며, 폐기 시 재활용이 가능하지만 현재 재활용 기술이 완벽하지 않은 문제가 있습니다.",
            "metadata": {
                "question": "태양광 패널의 수명과 재활용은 어떻게 되나요?",
                "source_url": "https://example.com/solar-panels",
                "category": "재생에너지",
            },
        },
        {
            "text": "전기차 배터리는 리튬, 코발트 등 희소 금속을 사용하며, 채굴 과정에서 환경 파괴와 인권 문제가 발생할 수 있습니다.",
            "metadata": {
                "question": "전기차가 정말 친환경적인가요?",
                "source_url": "https://example.com/ev-batteries",
                "category": "전기차",
            },
        },
    ]
    print("Test data prepared.\n")

    # Step 3: Store test data
    print("[3/5] Storing test data...")
    try:
        for doc in test_documents:
            doc_id = str(uuid.uuid4())
            success = redis_handler.save_embedding(
                key=doc_id, text=doc["text"], metadata=doc["metadata"]
            )
            if not success:
                print(f"Document save failed: {doc_id}")
                return
        print("Test data stored.\n")
    except Exception as e:
        print(f"Data storage failed: {e}")
        return

    # Step 4: Run similarity searches
    print("[4/5] Running similarity searches...")
    try:
        test_queries = [
            "종이 빨대의 환경적 영향은 어떤가요?",
            "태양광 패널의 재활용 현황은 어떻게 되나요?",
            "전기차 배터리의 환경적 문제점은 무엇인가요?",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            results = redis_handler.search_similar_embeddings(
                query_text=query, top_k=2, similarity_threshold=0.4
            )

            if results:
                print(f"Results ({len(results)}):")
                for idx, result in enumerate(results, 1):
                    print(f"  [{idx}] similarity: {result['similarity']:.4f}")
                    print(f"      question: {result['metadata'].get('question', 'N/A')}")
                    print(f"      text: {result['metadata'].get('text', 'N/A')[:100]}...")
            else:
                print("No results found.")

        print("\nSimilarity search test complete.\n")
    except Exception as e:
        print(f"Search test failed: {e}")
        return

    # Step 5: Display index info
    print("[5/5] Retrieving index info...")
    try:
        index_info = redis_handler.get_index_info()
        print("\nIndex info:")
        for key, value in index_info.items():
            print(f"  {key}: {value}")
        print("\nIndex info retrieval complete.\n")
    except Exception as e:
        print(f"Index info retrieval failed: {e}")
        return

    print("=== Redis 8 Vector Search Test Complete ===\n")


if __name__ == "__main__":
    test_vector_search()
