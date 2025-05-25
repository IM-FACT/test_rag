# Test RAG System

환경 및 기후 관련 질문에 대한 답변을 효율적으로 처리하기 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 📋 목차
- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 구조](#시스템-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [API 문서](#api-문서)
- [설정](#설정)
- [트러블슈팅](#트러블슈팅)

## 개요

이 시스템은 사용자의 환경/기후 관련 질문에 대해:
1. 기존에 유사한 질문이 있었는지 검색
2. 유사한 질문이 있으면 캐시된 답변 활용
3. 없으면 새로운 답변을 생성하고 저장

이를 통해 중복 질문에 대한 응답 속도를 향상시키고 API 비용을 절감합니다.

## 주요 기능

### 1. 임베딩 생성 (Embedding Generation)
- OpenAI의 `text-embedding-3-small` 모델 사용
- 텍스트를 고차원 벡터로 변환
- 의미적 유사도 계산을 위한 기반 제공

### 2. 유사도 검색 (Similarity Search)
- 코사인 유사도 기반 벡터 검색
- 설정 가능한 유사도 임계값 (기본: 0.6)
- Top-K 결과 반환 지원

### 3. Redis 기반 벡터 저장소
- 임베딩 벡터와 메타데이터 영구 저장
- 빠른 검색을 위한 인덱싱
- 확장 가능한 구조

## 시스템 구조

```
test_rag/
├── rag/                      # RAG 시스템 핵심 모듈
│   ├── embedding_generator.py    # 텍스트 임베딩 생성
│   ├── redis_handler.py          # Redis 벡터 DB 관리
│   └── main_processor.py         # 메인 처리 로직
├── langchain/                # LangChain 버전 (동일 구조)
│   └── ...
├── requirements.txt          # 필요 패키지 목록
├── .env                     # 환경 변수 (OPENAI_API_KEY)
└── README.md               # 프로젝트 문서
```

### 핵심 컴포넌트

#### 1. EmbeddingGenerator (`embedding_generator.py`)
```python
# 주요 기능:
- OpenAI API를 통한 텍스트 임베딩 생성
- 1536차원 벡터 출력 (text-embedding-3-small)
- 에러 처리 및 유효성 검증
```

#### 2. RedisHandler (`redis_handler.py`)
```python
# 주요 기능:
- Redis 연결 관리
- 임베딩 벡터 저장/조회
- 코사인 유사도 계산
- 유사 임베딩 검색
```

#### 3. MainProcessor (`main_processor.py`)
```python
# 주요 기능:
- 전체 프로세스 조율
- 임베딩 생성 → 유사도 검색 → 저장/반환
- 결과 포맷팅 및 출력
```

## 설치 방법

### 1. 필수 요구사항
- Python 3.8 이상
- Redis 서버 (로컬 또는 원격)
- OpenAI API 키

### 2. 패키지 설치
```bash
cd test_rag
pip install -r requirements.txt
```

### 3. 환경 변수 설정
`.env` 파일 생성:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 4. Redis 서버 실행
```bash
# Docker를 사용하는 경우
docker run -d -p 6379:6379 redis:latest

# 또는 로컬 설치
redis-server
```

## 사용 방법

### 기본 사용법

```python
from rag.main_processor import MainProcessor

# 프로세서 초기화
processor = MainProcessor(redis_host='localhost', redis_port=6379)

# 질문 처리
result = processor.process(
    query="종이 빨대가 정말 환경에 도움이 되나요?",
    source="https://example.com/article",
    text="종이 빨대의 환경적 영향에 대한 분석..."
)

# 결과 출력
processor.display_results(result)
```

### 직접 실행
```bash
python rag/main_processor.py
```

### 반환 결과 구조

```python
{
    "success": bool,           # 처리 성공 여부
    "operation": str,          # "found_similar" 또는 "saved_new"
    "message": str,           # 결과 메시지
    "similar_items": [        # 유사한 항목들 (있는 경우)
        {
            "key": str,
            "similarity": float,
            "metadata": {
                "question": str,
                "source_url": str,
                "timestamp": float,
                "text": str
            }
        }
    ]
}
```

## API 문서

### EmbeddingGenerator

#### `generate_embedding(text: str) -> Optional[List[float]]`
텍스트를 임베딩 벡터로 변환합니다.

**매개변수:**
- `text`: 임베딩할 텍스트

**반환값:**
- 성공 시: 1536차원 float 리스트
- 실패 시: None

### RedisHandler

#### `save_embedding(key: str, embedding: List[float], metadata: Dict) -> bool`
임베딩과 메타데이터를 Redis에 저장합니다.

#### `search_similar_embeddings(query_embedding: List[float], top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]`
유사한 임베딩을 검색합니다.

**매개변수:**
- `query_embedding`: 검색할 임베딩 벡터
- `top_k`: 반환할 최대 결과 수
- `similarity_threshold`: 유사도 임계값

### MainProcessor

#### `process(query: str, source: str, text: str) -> Dict`
전체 RAG 파이프라인을 실행합니다.

## 설정

### 유사도 임계값 조정
`main_processor.py`에서:
```python
SIMILARITY_THRESHOLD = 0.6  # 0.0 ~ 1.0 (높을수록 엄격)
```

### Redis 연결 설정
```python
processor = MainProcessor(
    redis_host='your-redis-host',
    redis_port=6379
)
```

## 트러블슈팅

### 1. Redis 연결 오류
```
Redis 연결 오류: Connection refused
```
**해결:** Redis 서버가 실행 중인지 확인하고, 호스트와 포트가 올바른지 확인

### 2. OpenAI API 키 오류
```
오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.
```
**해결:** `.env` 파일에 유효한 API 키 설정

### 3. 임베딩 생성 실패
```
임베딩 생성 오류: Invalid API key
```
**해결:** OpenAI API 키가 유효하고 크레딧이 있는지 확인

## 향후 개선 사항

1. **성능 최적화**
   - 배치 임베딩 처리
   - 비동기 처리 지원
   - 캐시 만료 정책

2. **기능 확장**
   - 다양한 임베딩 모델 지원
   - 하이브리드 검색 (키워드 + 벡터)
   - 메타데이터 필터링

3. **모니터링**
   - 검색 성능 메트릭
   - 캐시 히트율 추적
   - API 사용량 모니터링

## 라이선스

이 프로젝트는 IM.FACT 시스템의 일부입니다.
