"""Embedding generator module using LangChain and OpenAI.

Generates text embeddings with Redis-based caching to avoid redundant
API calls for previously embedded texts.
"""

import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.redis_handler import EmbeddingsCacheHandler
import numpy as np


class EmbeddingGenerator:
    """Generates text embeddings via OpenAI API with Redis caching.

    Attributes:
        model_name: Name of the OpenAI embedding model.
        embeddings: LangChain OpenAI embeddings instance.
        cache: Redis-based embedding cache handler.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        redis_url: str = "redis://localhost:6379",
    ):
        """Initializes the embedding generator.

        Args:
            model_name: OpenAI embedding model name.
            redis_url: Redis server URL.
        """
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

        if not os.path.exists(env_path):
            print(f"Error: .env file not found at {env_path}")
            sys.exit(1)

        load_dotenv(env_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Error: OPENAI_API_KEY environment variable is not set.")
            sys.exit(1)

        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(
            model=model_name, openai_api_key=self.api_key
        )
        self.cache = EmbeddingsCacheHandler(redis_url=redis_url)

        print(f"EmbeddingGenerator initialized: model={model_name}")

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Converts text into an embedding vector with cache lookup.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a numpy array, or None on failure.
        """
        if not text or text.strip() == "":
            print("Error: empty text provided for embedding.")
            return None
        try:
            cached = self.cache.get_embedding(text)
            if cached is not None:
                return cached

            embedding = self.embeddings.embed_query(text)
            embedding_np = np.array(embedding, dtype=np.float32)
            self.cache.set_embedding(text, embedding_np)
            return embedding_np
        except Exception as e:
            print(f"Embedding generation error: {e}")
            return None
