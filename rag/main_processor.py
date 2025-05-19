import sys
import uuid
from typing import Dict, Any, List
import time

# 로컬 모듈 임포트
from embedding_generator import EmbeddingGenerator
from redis_handler import RedisHandler

# 개발자 수정 가능 변수 (예시)
user_query = "요즘 카페에서 다 종이 빨대 쓰는데, 이게 진짜 지구온난화 막는 데 의미가 있어? 세척도 안돼서 결국 일반쓰레기고 다 태우면 탄소 배출도 그대로고, 여러 개 쓰게 되면 더 낭비 아닌가?"
source_link = "https://www.hani.co.kr/arti/society/environment/1156942.html"
text_to_embed = "종이 빨대에 플라스틱 코팅을 사용하는 이유와 그로 인한 단점은 뭔가요?"

# 유사도 임계값 (이 값 이상의 유사도를 가진 결과가 있으면 유사한 것으로 간주)
SIMILARITY_THRESHOLD = 0.6

class MainProcessor:
    """RAG 시스템의 메인 처리 로직을 담당하는 클래스"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        """
        메인 프로세서 초기화
        
        Args:
            redis_host (str): Redis 서버 호스트명
            redis_port (int): Redis 서버 포트
        """
        try:
            # 임베딩 생성기 초기화
            self.embedding_generator = EmbeddingGenerator()
            
            # Redis 핸들러 초기화
            self.redis_handler = RedisHandler(host=redis_host, port=redis_port)
            
            print("메인 프로세서 초기화 완료")
        except Exception as e:
            print(f"메인 프로세서 초기화 오류: {e}")
            sys.exit(1)

    def process(self, query: str, source: str, text: str) -> Dict[str, Any]:
        """
        주어진 텍스트를 임베딩하고 유사도 검색 또는 저장을 수행
        
        Args:
            query (str): 사용자 질문
            source (str): 정보 출처 링크
            text (str): 임베딩할 텍스트
            
        Returns:
            Dict[str, Any]: 처리 결과 정보
        """
        result = {
            "success": False,
            "operation": None,
            "message": "",
            "similar_items": []
        }
        
        # 텍스트 임베딩 생성
        print(f"[1/3] 텍스트 임베딩 생성 중...")
        embedding = self.embedding_generator.generate_embedding(text)
        
        if not embedding:
            result["message"] = "임베딩 생성에 실패했습니다."
            return result
            
        print(f"[1/3] 임베딩 생성 완료: {len(embedding)} 차원 벡터")
        
        # 유사한 임베딩 검색
        print(f"[2/3] 유사 임베딩 검색 중...")
        similar_items = self.redis_handler.search_similar_embeddings(
            query_embedding=embedding,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        
        # 처리 결과에 따라 분기
        if similar_items:
            # 유사한 항목이 존재하는 경우
            print(f"[3/3] 유사한 항목 {len(similar_items)}개 발견")
            result["operation"] = "found_similar"
            result["similar_items"] = similar_items
            result["success"] = True
            result["message"] = f"{len(similar_items)}개의 유사한 항목을 찾았습니다."
        else:
            # 유사한 항목이 없는 경우 -> 새로 저장
            print(f"[3/3] 유사한 항목 없음: 새 임베딩 저장 중...")
            
            # 메타데이터 생성
            metadata = {
                "question": query,
                "source_url": source,
                "timestamp": time.time(),
                "text": text
            }
            
            # 고유 키 생성 (UUID)
            unique_key = str(uuid.uuid4())
            
            # Redis에 저장
            save_result = self.redis_handler.save_embedding(
                key=unique_key,
                embedding=embedding,
                metadata=metadata
            )
            
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
                
                print(f"[{idx}] 유사도: {similarity:.4f} ({similarity*100:.1f}%)")
                print(f"    질문: {metadata.get('question', 'N/A')}")
                print(f"    출처: {metadata.get('source_url', 'N/A')}")
                print(f"    내용: {metadata.get('text', 'N/A')[:100]}...")
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
    print(f"출처: {source_link}")
    print(f"텍스트: {text_to_embed[:50]}...")
    print("=" * 80)
    
    # 처리 실행
    result = processor.process(
        query=user_query,
        source=source_link,
        text=text_to_embed
    )
    
    # 결과 출력
    processor.display_results(result)

if __name__ == "__main__":
    main()
