import redis
from langchain.vector_search import VectorSearchIndex
import numpy as np
from typing import List, Dict, Any
import time
import uuid
import hashlib


def get_redis_client(redis_url: str) -> redis.Redis:
    """
    Redis 클라이언트 생성 유틸 함수 (decode_responses=False 고정)
    """
    return redis.Redis.from_url(redis_url, decode_responses=False)


class RedisVectorSearchHandler:
    """Redis 8 Vector Search를 활용한 핸들러"""
    
    def __init__(self, 
                 embedding_model,
                 redis_url: str = "redis://localhost:6379",
                 index_name: str = "document_index"):
        """
        Redis Vector Search 핸들러 초기화
        
        Args:
            embedding_model: LangChain 임베딩 모델
            redis_url: Redis 서버 URL
            index_name: 벡터 검색 인덱스 이름
        """
        try:
            self.embedding_model = embedding_model
            self.redis_url = redis_url
            self.index_name = index_name
            
            # Redis 클라이언트 생성 (유틸 함수 사용)
            self.redis_client = get_redis_client(redis_url)
            
            # Vector Search 인덱스 초기화
            self.vector_index = VectorSearchIndex(
                redis_client=self.redis_client,
                index_name=index_name,
                vector_dimension=1536,  # OpenAI text-embedding-3-small
                distance_metric="COSINE"
            )
            
            print(f"Redis Vector Search 핸들러 초기화 완료: {redis_url}")
            
        except Exception as e:
            print(f"Redis Vector Search 핸들러 초기화 오류: {e}")
            raise
    
    def save_embedding(self, key: str, text: str, metadata: dict) -> bool:
        """
        텍스트와 메타데이터를 임베딩하여 Vector Search 인덱스에 저장
        
        Args:
            key: 문서 고유 키
            text: 임베딩할 텍스트
            metadata: 저장할 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 텍스트를 임베딩 벡터로 변환
            embedding = self.embedding_model.embed_query(text)
            
            # 메타데이터 준비
            doc_metadata = metadata.copy()
            doc_metadata["text"] = text
            doc_metadata["timestamp"] = doc_metadata.get("timestamp", time.time())
            
            # Vector Search 인덱스에 추가
            success = self.vector_index.add_document(
                doc_id=key,
                embedding=embedding,
                metadata=doc_metadata
            )
            
            if success:
                print(f"문서 저장 완료 - ID: {key}")
            
            return success
            
        except Exception as e:
            print(f"임베딩 저장 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_similar_embeddings(self, 
                                 query_text: str,
                                 top_k: int = 5,
                                 similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        텍스트 쿼리로 유사한 문서 검색
        
        Args:
            query_text: 검색할 텍스트
            top_k: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        try:
            # 쿼리 텍스트를 임베딩으로 변환
            query_embedding = self.embedding_model.embed_query(query_text)
            
            # Vector Search로 유사 문서 검색
            results = self.vector_index.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=similarity_threshold
            )
            
            print(f"검색 완료: {len(results)}개 결과 (임계값: {similarity_threshold})")
            
            return results
            
        except Exception as e:
            print(f"유사 임베딩 검색 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def delete_embedding(self, key: str) -> bool:
        """
        저장된 임베딩 삭제
        
        Args:
            key: 삭제할 문서의 키
            
        Returns:
            bool: 삭제 성공 여부
        """
        return self.vector_index.delete_document(key)
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        인덱스 정보 조회
        
        Returns:
            Dict: 인덱스 정보
        """
        return self.vector_index.get_index_info()

    def get_all_stored_documents(self) -> list:
        """Redis에 저장된 모든 문서를 조회"""
        try:
            all_keys = self.redis_client.keys(f"doc:{self.index_name}:*".encode('utf-8'))

            all_documents = []
            for key in all_keys:
                try:
                    key_str = key.decode('utf-8')
                    hash_data = self.redis_client.hgetall(key)

                    metadata = {}
                    text_content = None

                    for field, value in hash_data.items():
                        try:
                            field_str = field.decode('utf-8') if isinstance(field, bytes) else str(field)

                            # 임베딩 벡터 필드들은 제외 (바이너리 데이터)
                            if field_str in ['embedding_vector', 'content_vector']:
                                continue

                            # UTF-8 디코딩 시도
                            if isinstance(value, bytes):
                                try:
                                    value_str = value.decode('utf-8')
                                except UnicodeDecodeError:
                                    # 디코딩 실패시 건너뛰기
                                    continue
                            else:
                                value_str = str(value)

                            if field_str == 'text':
                                text_content = value_str
                            else:
                                metadata[field_str] = value_str

                        except Exception as e:
                            # 개별 필드 처리 오류는 무시하고 계속 진행
                            continue

                    document_info = {
                        "redis_key": key_str,
                        "key": metadata.get("custom_key", metadata.get("id", "unknown")),
                        "metadata": metadata,
                        "text": text_content or "N/A"
                    }

                    all_documents.append(document_info)

                except Exception as e:
                    print(f"문서 처리 오류 ({key}): {e}")
                    continue

            return all_documents

        except Exception as e:
            print(f"전체 문서 조회 오류: {e}")
            return []

class SemanticCacheHandler:
    """
    Redis 8 기반 시멘틱 캐시 핸들러 (질문-답변 쌍, 벡터 유사도 기반)
    """
    def __init__(self, embedding_model, redis_url: str = "redis://localhost:6379", index_name: str = "semantic_cache_index"):
        self.embedding_model = embedding_model
        self.redis_url = redis_url
        self.index_name = index_name
        self.redis_client = get_redis_client(redis_url)
        self.vector_index = VectorSearchIndex(
            redis_client=self.redis_client,
            index_name=index_name,
            vector_dimension=1536,  # OpenAI text-embedding-3-small
            distance_metric="COSINE"
        )

    def save_qa_pair(self, question: str, answer: str, metadata: dict = None) -> bool:
        """
        질문-답변 쌍을 임베딩하여 벡터 인덱스에 저장
        """
        try:
            embedding = self.embedding_model.embed_query(question)
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata["question"] = question
            doc_metadata["answer"] = answer
            doc_metadata["timestamp"] = doc_metadata.get("timestamp", time.time())
            doc_metadata["type"] = "semantic_cache"
            key = str(uuid.uuid4())
            return self.vector_index.add_document(
                doc_id=key,
                embedding=embedding,
                metadata=doc_metadata
            )
        except Exception as e:
            print(f"[SemanticCache] 저장 오류: {e}")
            import traceback; traceback.print_exc()
            return False

    def search_similar_question(self, query: str, top_k: int = 3, score_threshold: float = 0.05):
        """
        쿼리와 유사한 질문-답변 쌍을 score_threshold 기준으로 검색
        """
        try:
            embedding = self.embedding_model.embed_query(query)
            results = self.vector_index.search_similar(
                query_vector=embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )
            # answer 필드만 추출
            return [
                {
                    "question": r["metadata"].get("question"),
                    "answer": r["metadata"].get("answer"),
                    "similarity": r["similarity"]
                }
                for r in results
            ]
        except Exception as e:
            print(f"[SemanticCache] 검색 오류: {e}")
            import traceback; traceback.print_exc()
            return []


class EmbeddingsCacheHandler:
    """
    Redis 기반 임베딩 캐시 핸들러 (텍스트-SHA256 해시를 key, 임베딩 bytes를 value)
    """
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "embeddings_cache"):
        self.redis_url = redis_url
        self.prefix = prefix
        self.redis_client = get_redis_client(redis_url)

    def _make_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.prefix}:{h}"

    def get_embedding(self, text: str):
        key = self._make_key(text)
        value = self.redis_client.get(key)
        if value is not None:
            return np.frombuffer(value, dtype=np.float32)
        return None

    def set_embedding(self, text: str, embedding: np.ndarray):
        key = self._make_key(text)
        self.redis_client.set(key, embedding.astype(np.float32).tobytes())
