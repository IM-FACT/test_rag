# embedding_generator.py (LangChain 버전)
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


class EmbeddingGenerator:
    """LangChain을 사용하여 텍스트 임베딩을 생성하는 클래스"""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        임베딩 생성기 초기화

        Args:
            model_name (str): 사용할 OpenAI 임베딩 모델명
        """
        # .env 파일 로드 (상위 디렉토리에 있는 .env 파일)
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

        if not os.path.exists(env_path):
            print(f"오류: .env 파일을 찾을 수 없습니다. 경로: {env_path}")
            sys.exit(1)

        load_dotenv(env_path)

        # API 키 확인
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            sys.exit(1)

        # 모델명 저장
        self.model_name = model_name

        # LangChain OpenAI 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=self.api_key
        )

        print(f"임베딩 생성기 초기화 완료: 모델 {model_name}")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            Optional[List[float]]: 임베딩 벡터 (실패 시 None)
        """
        if not text or text.strip() == "":
            print("오류: 임베딩할 텍스트가 비어 있습니다.")
            return None

        try:
            # LangChain을 사용하여 임베딩 생성
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return None