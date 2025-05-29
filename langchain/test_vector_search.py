# test_vector_search.py
"""
Redis 8 Vector Search 기능 테스트 스크립트
"""

import sys
import os
from embedding_generator import EmbeddingGenerator
from redis_handler import RedisVectorSearchHandler
import uuid
import time


def test_vector_search():
    """Vector Search 기능을 테스트하는 함수"""
    
    print("\n=== Redis 8 Vector Search 테스트 시작 ===\n")
    
    # 1. 컴포넌트 초기화
    print("[1/5] 컴포넌트 초기화 중...")
    try:
        # 임베딩 생성기 초기화
        embedding_gen = EmbeddingGenerator()
        
        # Redis Vector Search 핸들러 초기화
        redis_handler = RedisVectorSearchHandler(
            embedding_model=embedding_gen.embeddings,
            redis_url="redis://host.docker.internal:6379",
            index_name="climate_vectors_test"
        )
        print("✅ 컴포넌트 초기화 완료\n")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    # 2. 테스트 데이터 준비
    print("[2/5] 테스트 데이터 준비 중...")
    test_documents = [
        {
            "text": "종이 빨대는 플라스틱 빨대의 대안으로 사용되지만, 제조 과정에서 화학물질을 사용하고 재활용이 어려운 단점이 있습니다.",
            "metadata": {
                "question": "종이 빨대가 정말 친환경적인가요?",
                "source_url": "https://example.com/paper-straws",
                "category": "플라스틱 대체품"
            }
        },
        {
            "text": "태양광 패널은 일반적으로 25-30년의 수명을 가지며, 폐기 시 재활용이 가능하지만 현재 재활용 기술이 완벽하지 않은 문제가 있습니다.",
            "metadata": {
                "question": "태양광 패널의 수명과 재활용은 어떻게 되나요?",
                "source_url": "https://example.com/solar-panels",
                "category": "재생에너지"
            }
        },
        {
            "text": "전기차 배터리는 리튬, 코발트 등 희소 금속을 사용하며, 채굴 과정에서 환경 파괴와 인권 문제가 발생할 수 있습니다.",
            "metadata": {
                "question": "전기차가 정말 친환경적인가요?",
                "source_url": "https://example.com/ev-batteries",
                "category": "전기차"
            }
        }
    ]
    print("✅ 테스트 데이터 준비 완료\n")
    
    # 3. 테스트 데이터 저장
    print("[3/5] 테스트 데이터 저장 중...")
    try:
        for doc in test_documents:
            # 고유 키 생성
            doc_id = str(uuid.uuid4())
            
            # Redis에 저장
            success = redis_handler.save_embedding(
                key=doc_id,
                text=doc["text"],
                metadata=doc["metadata"]
            )
            
            if not success:
                print(f"❌ 문서 저장 실패: {doc_id}")
                return
                
        print("✅ 테스트 데이터 저장 완료\n")
    except Exception as e:
        print(f"❌ 데이터 저장 실패: {e}")
        return
    
    # 4. 유사도 검색 테스트
    print("[4/5] 유사도 검색 테스트 중...")
    try:
        # 테스트 쿼리
        test_queries = [
            "종이 빨대의 환경적 영향은 어떤가요?",
            "태양광 패널의 재활용 현황은 어떻게 되나요?",
            "전기차 배터리의 환경적 문제점은 무엇인가요?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 검색 쿼리: {query}")
            results = redis_handler.search_similar_embeddings(
                query_text=query,
                top_k=2,
                similarity_threshold=0.4
            )
            
            if results:
                print(f"✅ 검색 결과 ({len(results)}개):")
                for idx, result in enumerate(results, 1):
                    print(f"\n[{idx}] 유사도: {result['similarity']:.4f}")
                    print(f"    질문: {result['metadata'].get('question', 'N/A')}")
                    print(f"    내용: {result['metadata'].get('text', 'N/A')[:100]}...")
            else:
                print("❌ 검색 결과 없음")
                
        print("\n✅ 유사도 검색 테스트 완료\n")
    except Exception as e:
        print(f"❌ 검색 테스트 실패: {e}")
        return
    
    # 5. 인덱스 정보 확인
    print("[5/5] 인덱스 정보 확인 중...")
    try:
        index_info = redis_handler.get_index_info()
        print("\n📊 인덱스 정보:")
        for key, value in index_info.items():
            print(f"    {key}: {value}")
        print("\n✅ 인덱스 정보 확인 완료\n")
    except Exception as e:
        print(f"❌ 인덱스 정보 확인 실패: {e}")
        return
    
    print("=== Redis 8 Vector Search 테스트 완료 ===\n")


if __name__ == "__main__":
    test_vector_search() 