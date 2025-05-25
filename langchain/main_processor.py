# main_processor.py (LangChain 버전)
import sys
import uuid
from typing import Dict, Any, List
import time

# 로컬 모듈 임포트
from embedding_generator import EmbeddingGenerator
from redis_handler import RedisHandlerFixed

# 개발자 수정 가능 변수 (예시)
user_query = "종이 빨대에 플라스틱 코팅을 사용하는 이유와 그로 인한 단점은 뭔가요?"

# 유사도 임계값 (이 값 이상의 유사도를 가진 결과가 있으면 유사한 것으로 간주)
SIMILARITY_THRESHOLD = 0.4


class MainProcessor:
    """LangChain 기반 RAG 시스템의 메인 처리 로직을 담당하는 클래스"""

    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        """
        메인 프로세서 초기화

        Args:
            redis_url (str): Redis 서버 URL
        """
        try:
            # 임베딩 생성기 초기화
            self.embedding_generator = EmbeddingGenerator()

            # Redis 핸들러 초기화 (임베딩 모델 전달)
            self.redis_handler = RedisHandlerFixed(
                embedding_model=self.embedding_generator.embeddings,
                redis_url=redis_url
            )

            print("메인 프로세서 초기화 완료")
        except Exception as e:
            print(f"메인 프로세서 초기화 오류: {e}")
            sys.exit(1)

    def process(self, query: str) -> Dict[str, Any]:
        """
        주어진 텍스트를 임베딩하고 유사도 검색 또는 저장을 수행

        Args:
            query (str): 사용자 질문

        Returns:
            Dict[str, Any]: 처리 결과 정보
        """
        result = {
            "success": False,
            "operation": None,
            "message": "",
            "similar_items": []
        }

        # 유사한 임베딩 검색
        print(f"[1/2] 유사 임베딩 검색 중...")
        similar_items = self.redis_handler.search_similar_embeddings(
            query_text=query,  # 검색할 텍스트
            similarity_threshold=SIMILARITY_THRESHOLD
        )

        # 처리 결과에 따라 분기
        if similar_items:
            # 유사한 항목이 존재하는 경우
            print(f"[2/2] 유사한 항목 {len(similar_items)}개 발견")
            result["operation"] = "found_similar"
            result["similar_items"] = similar_items
            result["success"] = True
            result["message"] = f"{len(similar_items)}개의 유사한 항목을 찾았습니다."
        else:
            # 유사한 항목이 없는 경우 -> 새로 저장
            print(f"[2/2] 유사한 항목 없음: 새 임베딩 저장 중...")

            # 메타데이터 생성
            metadata = {
                "question": query,
                "timestamp": time.time()
            }

            # 고유 키 생성 (UUID)
            unique_key = str(uuid.uuid4())

            # Redis에 저장
            save_result = self.redis_handler.save_embedding(
                key=unique_key,
                text=query,  # 임베딩할 텍스트
                metadata=metadata
            )
            # print(f"저장 전 메타데이터: {metadata}")
            if save_result:
                result["operation"] = "saved_new"
                result["success"] = True
                result["message"] = f"새 임베딩이 성공적으로 저장되었습니다. (키: {unique_key})"
            else:
                result["message"] = "새 임베딩 저장에 실패했습니다."

        return result

    def display_results(self, result: Dict[str, Any]) -> None:
        """
        처리 결과를 콘솔에 출력

        Args:
            result (Dict[str, Any]): 처리 결과 정보
        """
        if not result["success"]:
            print(f"\n❌ 오류: {result['message']}")
            return

        if result["operation"] == "found_similar":
            print("\n🔍 유사한 정보 발견:")
            print("-" * 80)

            for idx, item in enumerate(result["similar_items"], 1):
                metadata = item["metadata"]
                similarity = item["similarity"]

                print(f"[{idx}] 유사도: {similarity:.4f} ({similarity * 100:.1f}%)")
                print(f"    질문: {metadata.get('question', 'N/A')}")
                print(f"    출처: {metadata.get('source_url', 'N/A')}")
                print(f"    내용: {item.get('text', 'N/A')[:100]}...")
                print("-" * 80)
        elif result["operation"] == "saved_new":
            print("\n💾 새 정보 저장 완료:")
            print("-" * 80)
            print(f"메시지: {result['message']}")
            print("-" * 80)


def main():
    """메인 실행 함수"""
    # 메인 프로세서 초기화
    processor = MainProcessor()

    # 현재 설정된 입력값 표시
    print("\n=== RAG 프로세서 실행 ===")
    print(f"질문: {user_query}")
    print("=" * 80)

    # 처리 실행
    result = processor.process(
        query=user_query,
    )

    # 결과 출력
    processor.display_results(result)


if __name__ == "__main__":
    main()