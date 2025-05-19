import redis
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

class RedisHandler:
    """Redis 연동을 위한 핸들러 클래스"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: str = None):
        """
        Redis 연결 설정
        
        Args:
            host (str): Redis 서버 호스트명
            port (int): Redis 서버 포트
            db (int): 사용할 Redis DB 번호
            password (str, optional): Redis 서버 비밀번호
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True  # 문자열 응답을 자동으로 디코딩
            )
            # 연결 테스트
            self.redis_client.ping()
            print(f"Redis 서버 연결 성공: {host}:{port}")
        except redis.ConnectionError as e:
            print(f"Redis 연결 오류: {e}")
            raise

    def save_embedding(self, key: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        임베딩 벡터와 메타데이터를 Redis에 저장
        
        Args:
            key (str): 저장할 데이터의 키
            embedding (List[float]): 임베딩 벡터
            metadata (Dict[str, Any]): 저장할 메타데이터 (질문, 출처 등)
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 메타데이터에 임베딩 벡터 추가
            data = {
                "embedding": json.dumps(embedding),  # 리스트를 JSON 문자열로 변환
                "metadata": json.dumps(metadata)     # 메타데이터를 JSON 문자열로 변환
            }
            
            # Redis Hash 구조로 저장
            self.redis_client.hset(f"embedding:{key}", mapping=data)
            return True
        except Exception as e:
            print(f"임베딩 저장 오류: {e}")
            return False

    def get_embedding(self, key: str) -> Tuple[Optional[List[float]], Optional[Dict[str, Any]]]:
        """
        저장된 임베딩 벡터와 메타데이터를 키를 통해 조회
        
        Args:
            key (str): 조회할 데이터의 키
            
        Returns:
            Tuple[Optional[List[float]], Optional[Dict[str, Any]]]: (임베딩 벡터, 메타데이터) 튜플
        """
        try:
            data = self.redis_client.hgetall(f"embedding:{key}")
            
            if not data:
                return None, None
                
            embedding = json.loads(data.get("embedding"))
            metadata = json.loads(data.get("metadata"))
            
            return embedding, metadata
        except Exception as e:
            print(f"임베딩 조회 오류: {e}")
            return None, None

    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        두 벡터 간의 코사인 유사도 계산
        
        Args:
            vec1 (List[float]): 첫 번째 벡터
            vec2 (List[float]): 두 번째 벡터
            
        Returns:
            float: 코사인 유사도 (0~1 사이의 값)
        """
        # NumPy 배열로 변환
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # 코사인 유사도 계산 (내적 / (노름 * 노름))
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # 0으로 나누기 방지
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return float(similarity)

    def search_similar_embeddings(self, query_embedding: List[float], 
                                top_k: int = 5, 
                                similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        쿼리 임베딩과 가장 유사한 임베딩 벡터와 메타데이터를 검색
        
        Args:
            query_embedding (List[float]): 검색할 쿼리 임베딩 벡터
            top_k (int): 반환할 가장 유사한 결과의 수
            similarity_threshold (float): 유사도 임계값 (이 값 이상의 유사도를 가진 결과만 반환)
            
        Returns:
            List[Dict[str, Any]]: 유사한 임베딩 및 메타데이터 목록
        """
        results = []
        
        try:
            # 저장된 모든 임베딩 키 조회
            embedding_keys = self.redis_client.keys("embedding:*")
            
            # 각 임베딩에 대해 유사도 계산
            similarities = []
            for key in embedding_keys:
                # 키에서 'embedding:' 접두사 제거
                key_id = key.replace("embedding:", "")
                
                # 임베딩 및 메타데이터 조회
                embedding, metadata = self.get_embedding(key_id)
                
                if embedding:
                    # 코사인 유사도 계산
                    similarity = self.compute_cosine_similarity(query_embedding, embedding)
                    
                    # 임계값 이상인 경우에만 결과에 추가
                    if similarity >= similarity_threshold:
                        similarities.append({
                            "key": key_id,
                            "similarity": similarity,
                            "embedding": embedding,
                            "metadata": metadata
                        })
            
            # 유사도 기준으로 내림차순 정렬
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # top_k개 결과 반환
            results = similarities[:top_k]
            
            return results
        except Exception as e:
            print(f"유사 임베딩 검색 오류: {e}")
            return []

    def delete_embedding(self, key: str) -> bool:
        """
        저장된 임베딩 삭제
        
        Args:
            key (str): 삭제할 데이터의 키
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            result = self.redis_client.delete(f"embedding:{key}")
            return result > 0
        except Exception as e:
            print(f"임베딩 삭제 오류: {e}")
            return False
