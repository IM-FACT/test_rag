# main_processor.py (LangChain 버전)
import sys
from typing import Dict, Any, List
import time

# import os

# current_dir = os.path.dirname(os.path.abspath(__file__))      # 현재 langchain 폴더
# project_root = os.path.abspath(os.path.join(current_dir, ".."))  # TEST_RAG
# sys.path.insert(0, project_root)

# 로컬 모듈 임포트
from langchain.embedding_generator import EmbeddingGenerator
from langchain.redis_handler import RedisVectorSearchHandler, SemanticCacheHandler

# 개발자 수정 가능 변수 (예시)
user_query = "종이 빨대에 플라스틱 코팅을 사용하는 이유와 그로 인한 단점은 뭔가요?"

# 유사도 임계값 (이 값 이상의 유사도를 가진 결과가 있으면 유사한 것으로 간주)
SIMILARITY_THRESHOLD = 0.4


class MainProcessor:
    """LangChain 기반 RAG 시스템의 메인 처리 로직을 담당하는 클래스 (리팩토링)"""

    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        """
        메인 프로세서 초기화

        Args:
            redis_url (str): Redis 서버 URL
        """
        try:
            # 임베딩 생성기 및 Redis 핸들러 초기화
            self.embedding_generator = EmbeddingGenerator()
            self.redis_handler = RedisVectorSearchHandler(
                embedding_model=self.embedding_generator.embeddings,
                redis_url=redis_url,
                index_name="document_index"
            )
            self.semantic_cache = SemanticCacheHandler(
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
            "similar_items": [],
            "cache_answer": None,
            "vector_search_results": [],
            "final_answer": None
        }
        print("hello")
        print(self.redis_handler.get_all_stored_documents())
        # 1. 시멘틱 캐시 검색
        cache_results = self.semantic_cache.search_similar_question(
            query=query,
            score_threshold=0.05
        )
        if cache_results:
            best = max(cache_results, key=lambda x: x["similarity"])
            result["operation"] = "cache_hit"
            result["cache_answer"] = best["answer"]
            result["success"] = True
            result["message"] = "시멘틱 캐시에서 답변을 반환했습니다."
            result["final_answer"] = best["answer"]
            return result

        # 2. 벡터 검색 (문서 기반 근거 탐색)
        vector_results = self.redis_handler.search_similar_embeddings(
            query_text=query,
            top_k=3,
            similarity_threshold=0.4
        )
        result["vector_search_results"] = vector_results

        # 3. 답변 생성 (여기선 임시 답변)
        generated_answer = f"[임시 답변] '{query}'에 대한 답변입니다."
        # 4. 캐시에 저장
        self.semantic_cache.save_qa_pair(
            question=query,
            answer=generated_answer,
            metadata={"source": "gpt", "timestamp": time.time()}
        )
        result["operation"] = "cache_miss_saved"
        result["cache_answer"] = generated_answer
        result["success"] = True
        result["message"] = "새 답변을 생성하여 시멘틱 캐시에 저장했습니다."
        result["final_answer"] = generated_answer
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

        if result["operation"] == "cache_hit":
            print("\n🔍 [시멘틱 캐시 HIT] 답변:")
            print("-" * 80)
            print(result["cache_answer"])
            print("-" * 80)
        elif result["operation"] == "cache_miss_saved":
            print("\n💾 [시멘틱 캐시 MISS] 새 답변 저장:")
            print("-" * 80)
            print(result["cache_answer"])
            print("-" * 80)
            print("\n[벡터 검색 결과]")
            for idx, item in enumerate(result["vector_search_results"], 1):
                print(f"{idx}. {item['metadata'].get('text', '')} (유사도: {item['similarity']:.2f})")
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