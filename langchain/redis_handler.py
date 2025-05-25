from langchain_community.vectorstores import Redis

import redis



class RedisHandlerFixed:
    """메타데이터 문제를 완전히 해결한 Redis 핸들러"""

    def __init__(self, embedding_model, redis_url: str = "redis://localhost:6379", index_name: str = "document_index"):
        """
        Redis 연결 설정 (메타데이터 문제 해결 버전)
        """
        try:
            self.embedding_model = embedding_model
            self.redis_url = redis_url
            self.index_name = index_name
            
            # 먼저 직접 Redis 연결 (벡터 존재 여부 확인용)
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
            
            # 기존 벡터 문서가 있는지 확인
            existing_docs = self._check_existing_documents()
            
            if existing_docs:
                # 기존 문서가 있으면 빈 텍스트로 벡터스토어 연결
                print(f"기존 문서 {len(existing_docs)}개 발견. 기존 벡터스토어에 연결합니다.")
                self.vector_store = Redis(
                    embedding=embedding_model,
                    redis_url=redis_url,
                    index_name=index_name
                )
            else:
                # 기존 문서가 없으면 초기화 문서와 함께 생성
                print("기존 문서가 없습니다. 초기화 문서를 추가하며 벡터스토어를 생성합니다.")
                self.vector_store = Redis.from_texts(
                    texts=["초기화 문서입니다."],
                    embedding=embedding_model,
                    redis_url=redis_url,
                    index_name=index_name
                )
            
            print(f"Redis 벡터 스토어 연결 성공: {redis_url}")

        except Exception as e:
            print(f"Redis 연결 오류: {e}")
            raise

    def _check_existing_documents(self) -> list:
        """
        Redis에 기존 벡터 문서가 있는지 확인
        
        Returns:
            list: 기존 문서 키 리스트 (비어있으면 빈 리스트)
        """
        try:
            # 해당 인덱스에 속한 문서 키들을 검색
            all_keys = self.redis_client.keys(f"doc:{self.index_name}:*".encode('utf-8'))
            
            # 초기화 문서는 제외하고 실제 사용자 문서만 카운트
            actual_docs = []
            for key in all_keys:
                try:
                    key_str = key.decode('utf-8')
                    # Redis에서 content 필드 확인
                    content = self.redis_client.hget(key, 'content')
                    if content:
                        content_str = content.decode('utf-8') if isinstance(content, bytes) else str(content)
                        # 초기화 문서가 아닌 경우만 실제 문서로 간주
                        if content_str != "초기화 문서입니다.":
                            actual_docs.append(key_str)
                except Exception as e:
                    print(f"키 검사 중 오류 ({key}): {e}")
                    continue
            
            print(f"기존 실제 문서 수: {len(actual_docs)}개")
            return actual_docs
            
        except Exception as e:
            print(f"기존 문서 확인 중 오류: {e}")
            return []

    def save_embedding(self, key: str, text: str, metadata: dict) -> bool:
        """
        텍스트와 메타데이터를 임베딩하여 Redis에 저장
        """
        try:
            # 메타데이터 처리
            processed_metadata = metadata.copy()
            processed_metadata["id"] = key
            processed_metadata["custom_key"] = key
            
            # 모든 값을 문자열로 변환
            for k, v in processed_metadata.items():
                processed_metadata[k] = str(v)
            
            print(f"저장할 메타데이터: {processed_metadata}")

            # LangChain을 통해 저장
            ids = self.vector_store.add_texts(
                texts=[text],
                metadatas=[processed_metadata]
            )

            print(f"저장 완료 - ID: {ids}")
            return len(ids) > 0
            
        except Exception as e:
            print(f"임베딩 저장 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_metadata_from_redis(self, doc_id: str) -> dict:
        """
        Redis에서 직접 메타데이터를 조회하는 헬퍼 메서드
        """
        try:
            # Redis Hash에서 메타데이터 직접 조회
            hash_data = self.redis_client.hgetall(doc_id.encode('utf-8'))
            
            metadata = {}
            for field, value in hash_data.items():
                try:
                    field_str = field.decode('utf-8') if isinstance(field, bytes) else str(field)
                    
                    # content_vector는 제외
                    if field_str == 'content_vector':
                        continue
                        
                    if isinstance(value, bytes):
                        try:
                            value_str = value.decode('utf-8')
                            metadata[field_str] = value_str
                        except UnicodeDecodeError:
                            # 바이너리 데이터는 건너뛰기
                            continue
                    else:
                        metadata[field_str] = str(value)
                        
                except Exception as e:
                    print(f"메타데이터 필드 처리 오류 ({field}): {e}")
                    continue
            
            return metadata
            
        except Exception as e:
            print(f"Redis 직접 메타데이터 조회 오류: {e}")
            return {}

    def search_similar_embeddings(self, query_text: str,
                                  top_k: int = 5,
                                  similarity_threshold: float = 0.7) -> list:
        """
        메타데이터 문제를 해결한 검색 메서드
        """
        try:
            # 1. LangChain을 통한 유사도 검색
            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query_text,
                k=top_k
            )

            print(f"검색된 원본 결과 수: {len(results)}")

            formatted_results = []
            for idx, (doc, score) in enumerate(results):
                # 테스팅용 디버깅

                print(f"\n=== 결과 {idx + 1} 디버깅 ===")
                print(f"Document content: {doc.page_content[:100]}...")
                print(f"LangChain metadata: {doc.metadata}")
                print(f"Score: {score}")

                # 유사도 임계값 확인
                if score >= similarity_threshold:
                    # Redis에서 직접 메타데이터 조회
                    doc_key = None
                    
                    # 먼저 LangChain에서 제공하는 ID 확인
                    langchain_id = doc.metadata.get('id')
                    doc_key = langchain_id
                    
                    # Redis에서 완전한 메타데이터 조회
                    if doc_key:
                        complete_metadata = self._get_metadata_from_redis(doc_key)
                        print(f"Redis에서 조회한 완전한 메타데이터: {complete_metadata}")
                    else:
                        complete_metadata = doc.metadata
                        print(f"⚠️ Redis 키를 찾을 수 없어 LangChain 메타데이터 사용")
                    
                    result_item = {
                        "key": complete_metadata.get("custom_key", complete_metadata.get("id", "unknown")),
                        "similarity": score,
                        "metadata": complete_metadata,
                        "text": doc.page_content,
                        "redis_key": doc_key
                    }
                    
                    print(f"최종 결과 item: {result_item}")
                    formatted_results.append(result_item)

            print(f"임계값 이상 결과 수: {len(formatted_results)}")
            return formatted_results
            
        except Exception as e:
            print(f"유사 임베딩 검색 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_stored_documents(self) -> list:
        """
        Redis에 저장된 모든 문서를 직접 조회
        """
        try:
            all_keys = self.redis_client.keys(f"doc:{self.index_name}:*".encode('utf-8'))
            
            all_documents = []
            for key in all_keys:
                try:
                    key_str = key.decode('utf-8')
                    metadata = self._get_metadata_from_redis(key_str)
                    
                    # content 필드에서 텍스트 내용 가져오기
                    text_content = metadata.pop('content', 'N/A')
                    
                    all_documents.append({
                        "redis_key": key_str,
                        "key": metadata.get("custom_key", metadata.get("id", "unknown")),
                        "metadata": metadata,
                        "text": text_content
                    })
                    
                except Exception as e:
                    print(f"문서 처리 오류 ({key}): {e}")
                    continue
            
            return all_documents
            
        except Exception as e:
            print(f"전체 문서 조회 오류: {e}")
            return []

    def delete_embedding(self, key: str) -> bool:
        """
        저장된 임베딩 삭제
        """
        try:
            # Redis 키를 찾아서 삭제
            all_keys = self.redis_client.keys(f"doc:{self.index_name}:*".encode('utf-8'))
            
            for redis_key in all_keys:
                try:
                    key_str = redis_key.decode('utf-8')
                    metadata = self._get_metadata_from_redis(key_str)
                    
                    # custom_key 또는 id가 일치하면 삭제
                    if (metadata.get("custom_key") == key or 
                        metadata.get("id") == key):
                        
                        self.redis_client.delete(redis_key)
                        print(f"삭제 성공: {key_str}")
                        return True
                        
                except Exception as e:
                    print(f"삭제 시도 오류 ({redis_key}): {e}")
                    continue
            
            print(f"삭제할 키를 찾을 수 없음: {key}")
            return False
            
        except Exception as e:
            print(f"임베딩 삭제 오류: {e}")
            return False
