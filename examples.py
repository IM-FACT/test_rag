# RAG 시스템 사용 예시

from rag.main_processor import MainProcessor

# 예시 1: 환경 관련 질문 처리
def example_environmental_question():
    """환경 관련 질문을 처리하는 예시"""
    processor = MainProcessor()
    
    # 사용자 질문
    questions = [
        {
            "query": "전기차가 정말 친환경적인가요?",
            "source": "https://www.nature.com/articles/s41558-021-01032-7",
            "text": "전기차의 생산부터 폐기까지 전 생애주기 탄소 배출량 분석"
        },
        {
            "query": "태양광 패널의 수명은 얼마나 되나요?",
            "source": "https://www.nrel.gov/analysis/life-cycle-assessment.html",
            "text": "태양광 패널은 일반적으로 25-30년의 수명을 가지며..."
        }
    ]
    
    for q in questions:
        print(f"\n질문: {q['query']}")
        result = processor.process(q['query'], q['source'], q['text'])
        processor.display_results(result)

# 예시 2: 배치 처리
def batch_processing_example():
    """여러 질문을 한번에 처리하는 예시"""
    processor = MainProcessor()
    
    # CSV나 데이터베이스에서 질문 목록을 가져온다고 가정
    batch_data = [
        # (질문, 출처, 텍스트) 튜플
        ("재활용이 정말 효과적인가요?", "https://example1.com", "재활용의 실제 효과..."),
        ("탄소 포집 기술의 현재 수준은?", "https://example2.com", "CCUS 기술 현황..."),
        ("도시 농업의 환경적 이점은?", "https://example3.com", "도시 농업의 장점...")
    ]
    
    results = []
    for query, source, text in batch_data:
        result = processor.process(query, source, text)
        results.append({
            "query": query,
            "operation": result["operation"],
            "similar_count": len(result["similar_items"])
        })
    
    # 결과 요약
    print("\n=== 배치 처리 결과 ===")
    for r in results:
        print(f"질문: {r['query']}")
        print(f"  - 작업: {r['operation']}")
        print(f"  - 유사 항목 수: {r['similar_count']}")

# 예시 3: 유사도 임계값 실험
def similarity_threshold_experiment():
    """다양한 유사도 임계값으로 실험하는 예시"""
    from rag.redis_handler import RedisHandler
    from rag.embedding_generator import EmbeddingGenerator
    
    # 컴포넌트 초기화
    redis_handler = RedisHandler()
    embedding_gen = EmbeddingGenerator()
    
    # 테스트 질문
    test_query = "플라스틱 재활용의 실제 효율성은 어떻게 되나요?"
    query_embedding = embedding_gen.generate_embedding(test_query)
    
    # 다양한 임계값으로 검색
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n테스트 질문: {test_query}")
    print("=" * 60)
    
    for threshold in thresholds:
        results = redis_handler.search_similar_embeddings(
            query_embedding=query_embedding,
            similarity_threshold=threshold,
            top_k=10
        )
        
        print(f"\n임계값 {threshold}: {len(results)}개 결과")
        if results:
            top_similarity = results[0]['similarity']
            print(f"  최고 유사도: {top_similarity:.4f}")
            print(f"  최고 유사 질문: {results[0]['metadata']['question'][:50]}...")

# 예시 4: 커스텀 메타데이터 활용
def custom_metadata_example():
    """추가 메타데이터를 활용하는 예시"""
    from rag.redis_handler import RedisHandler
    from rag.embedding_generator import EmbeddingGenerator
    import uuid
    import time
    
    redis_handler = RedisHandler()
    embedding_gen = EmbeddingGenerator()
    
    # 확장된 메타데이터 구조
    enhanced_metadata = {
        "question": "바이오 플라스틱은 정말 분해가 되나요?",
        "source_url": "https://www.science.org/doi/10.1126/science.aba9475",
        "timestamp": time.time(),
        "text": "바이오 플라스틱의 생분해 조건과 실제 분해율...",
        # 추가 메타데이터
        "category": "플라스틱",
        "reliability_score": 0.95,
        "language": "ko",
        "keywords": ["바이오플라스틱", "생분해", "환경"],
        "author": "김환경 박사",
        "publication_date": "2024-01-15"
    }
    
    # 임베딩 생성 및 저장
    embedding = embedding_gen.generate_embedding(enhanced_metadata["text"])
    key = str(uuid.uuid4())
    
    success = redis_handler.save_embedding(key, embedding, enhanced_metadata)
    
    if success:
        print(f"확장된 메타데이터와 함께 저장 성공!")
        print(f"저장된 키: {key}")
        
        # 저장된 데이터 확인
        retrieved_embedding, retrieved_metadata = redis_handler.get_embedding(key)
        print(f"\n저장된 메타데이터:")
        for k, v in retrieved_metadata.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    # 원하는 예시 실행
    print("=== RAG 시스템 사용 예시 ===\n")
    
    # 1. 기본 환경 질문 처리
    example_environmental_question()
    
    # 2. 배치 처리
    # batch_processing_example()
    
    # 3. 유사도 임계값 실험
    # similarity_threshold_experiment()
    
    # 4. 커스텀 메타데이터
    # custom_metadata_example()
