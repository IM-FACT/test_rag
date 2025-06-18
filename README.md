# Test RAG System

환경 및 기후 관련 질문에 대한 답변을 효율적으로 처리하기 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 📋 목차

- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [실행 방법](#실행-방법)
- [테스트 방법](#테스트-방법)
- [트러블슈팅](#트러블슈팅)

## 프로젝트 구조

```
test_rag/
├── langchain/ # 핵심 서비스 코드 (임베딩, 벡터 검색, 캐시 등)
│ ├── embedding_generator.py # OpenAI 임베딩
│ ├── redis_handler.py 
│ ├── vector_search.py
│ └── main_processor.py 
│
├── scrap_mcp/ # 문서 수집 및 GPT 기반 답변 생성 모듈
│ ├── mcp_module.py # Brave Search + 본문 수집 + GPT 답변 통합
│ ├── main.py # 단독 실행용 스크래퍼
│ ├── prompts/ # GPT용 프롬프트
│ │ ├── generate_ans_prompt.txt
│ │ └── rewrite_query_prompt.txt
│ ├── brave_search_module/
│ │ ├── brave_search_impl.py
│ │ └── brave_search_test.py
│ ├── tool/
│ │ ├── gen_ans.py # GPT-4-turbo 답변 생성
│ │ ├── rewrite_query.py # GPT-4o 키워드 리라이팅
│ │ ├── bing.py
│ │ ├── goo_api.py
│ │ └── text.py
│ └── tests/
│   └── test_mcp_module.py
│
├── test/ # 테스트 코드 (테스트 전용)
│ ├── test_embedding_cache.py
│ ├── test_semantic_cache.py
│ └── test_vector_search.py
├── requirements.txt
├── .env.example
└── README.md
```

## 설치 방법

1. Python 3.8 이상 설치
2. Redis 8/Redis Stack 서버 준비 (Search/Vector 기능 필수)
3. 패키지 설치
   ```bash
   cd test_rag
   pip install -r requirements.txt
   ```
4. .env 파일 생성 및 API 키 입력
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   BRAVE_AI_API_KEY=your-brave-search-api-key-here
   GOOGLE_API_KEY=your-google-api-key-here
   ```
5. Redis Stack 실행
   ```bash
   docker run -d -p 6379:6379 redis:latest
   # 또는 공식 문서 참고: https://redis.io/docs/stack/get-started/install/
   ```

## 실행 방법

서비스 코드 실행:

```bash
python langchain/main_processor.py
```

## 테스트 방법

테스트 코드는 test/ 폴더에 위치합니다. 아래와 같이 실행하세요:

```bash
python -m test.test_embedding_cache
python -m test.test_semantic_cache
python -m test.test_vector_search
```

> **TIP:** import 오류 없이 동작하려면, test_rag/ 루트에서 실행하거나 PYTHONPATH를 루트로 지정하세요.

## 트러블슈팅

- **Redis 연결 오류**: Redis Stack이 정상적으로 실행 중인지, 포트/호스트가 올바른지 확인하세요.
- **OpenAI API 키 오류**: .env 파일에 올바른 키가 입력되어 있는지 확인하세요.
- **임베딩 생성 실패**: OpenAI API 키가 유효하고 크레딧이 충분한지 확인하세요.
- **import 오류**: test_rag/ 루트에서 실행하거나 PYTHONPATH를 루트로 지정하세요.

## 라이선스

이 프로젝트는 IM.FACT 시스템의 일부이며, 별도의 라이선스 정책이 적용될 수 있습니다. 자세한 사항은 운영팀에 문의하세요.
