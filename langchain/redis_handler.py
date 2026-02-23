"""Redis handlers for vector search, semantic caching, and embedding caching.

Provides three handler classes:
- RedisVectorSearchHandler: stores and searches document embeddings.
- SemanticCacheHandler: caches question-answer pairs with vector similarity.
- EmbeddingsCacheHandler: caches raw embedding vectors by text hash.
"""

import redis
from langchain.vector_search import VectorSearchIndex
import numpy as np
from typing import List, Dict, Any
import time
import uuid
import hashlib


def get_redis_client(redis_url: str) -> redis.Redis:
    """Creates a Redis client with raw byte responses.

    Args:
        redis_url: Redis server URL.

    Returns:
        A Redis client instance with decode_responses=False.
    """
    return redis.Redis.from_url(redis_url, decode_responses=False)


class RedisVectorSearchHandler:
    """Handles document embedding storage and similarity search via Redis.

    Attributes:
        embedding_model: LangChain embedding model for vectorization.
        vector_index: Underlying VectorSearchIndex instance.
    """

    def __init__(
        self,
        embedding_model,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "document_index",
    ):
        """Initializes the vector search handler.

        Args:
            embedding_model: LangChain embedding model.
            redis_url: Redis server URL.
            index_name: Name of the vector search index.
        """
        try:
            self.embedding_model = embedding_model
            self.redis_url = redis_url
            self.index_name = index_name

            self.redis_client = get_redis_client(redis_url)
            self.vector_index = VectorSearchIndex(
                redis_client=self.redis_client,
                index_name=index_name,
                vector_dimension=1536,
                distance_metric="COSINE",
            )
            print(f"RedisVectorSearchHandler initialized: {redis_url}")
        except Exception as e:
            print(f"RedisVectorSearchHandler initialization error: {e}")
            raise

    def save_embedding(self, key: str, text: str, metadata: dict) -> bool:
        """Embeds text and stores it in the vector search index.

        Args:
            key: Unique document key.
            text: Text to embed.
            metadata: Document metadata to store alongside the vector.

        Returns:
            True if the document was saved successfully.
        """
        try:
            embedding = self.embedding_model.embed_query(text)

            doc_metadata = metadata.copy()
            doc_metadata["text"] = text
            doc_metadata["timestamp"] = doc_metadata.get("timestamp", time.time())

            success = self.vector_index.add_document(
                doc_id=key, embedding=embedding, metadata=doc_metadata
            )
            if success:
                print(f"Document saved - ID: {key}")
            return success
        except Exception as e:
            print(f"Embedding save error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def search_similar_embeddings(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Searches for documents similar to the given query text.

        Args:
            query_text: Text query to search for.
            top_k: Maximum number of results to return.
            similarity_threshold: Minimum cosine similarity threshold.

        Returns:
            List of result dicts with similarity scores and metadata.
        """
        try:
            query_embedding = self.embedding_model.embed_query(query_text)

            results = self.vector_index.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=similarity_threshold,
            )
            print(
                f"Search complete: {len(results)} results "
                f"(threshold: {similarity_threshold})"
            )
            return results
        except Exception as e:
            print(f"Similar embedding search error: {e}")
            import traceback

            traceback.print_exc()
            return []

    def delete_embedding(self, key: str) -> bool:
        """Deletes a stored embedding by key.

        Args:
            key: Document key to delete.

        Returns:
            True if the document was deleted successfully.
        """
        return self.vector_index.delete_document(key)

    def get_index_info(self) -> Dict[str, Any]:
        """Returns metadata about the vector search index."""
        return self.vector_index.get_index_info()

    def get_all_stored_documents(self) -> list:
        """Retrieves all documents stored in the index.

        Returns:
            List of document dicts with redis_key, key, metadata, and text.
        """
        try:
            all_keys = self.redis_client.keys(
                f"doc:{self.index_name}:*".encode("utf-8")
            )

            all_documents = []
            for key in all_keys:
                try:
                    key_str = key.decode("utf-8")
                    hash_data = self.redis_client.hgetall(key)

                    metadata = {}
                    text_content = None

                    for field, value in hash_data.items():
                        try:
                            field_str = (
                                field.decode("utf-8")
                                if isinstance(field, bytes)
                                else str(field)
                            )

                            if field_str in ["embedding_vector", "content_vector"]:
                                continue

                            if isinstance(value, bytes):
                                try:
                                    value_str = value.decode("utf-8")
                                except UnicodeDecodeError:
                                    continue
                            else:
                                value_str = str(value)

                            if field_str == "text":
                                text_content = value_str
                            else:
                                metadata[field_str] = value_str
                        except Exception:
                            continue

                    document_info = {
                        "redis_key": key_str,
                        "key": metadata.get(
                            "custom_key", metadata.get("id", "unknown")
                        ),
                        "metadata": metadata,
                        "text": text_content or "N/A",
                    }
                    all_documents.append(document_info)
                except Exception as e:
                    print(f"Document processing error ({key}): {e}")
                    continue

            return all_documents
        except Exception as e:
            print(f"All documents retrieval error: {e}")
            return []


class SemanticCacheHandler:
    """Caches question-answer pairs with vector similarity search.

    Uses Redis vector search to find semantically similar previously-answered
    questions, enabling cache hits for paraphrased queries.

    Attributes:
        embedding_model: LangChain embedding model.
        vector_index: Underlying VectorSearchIndex instance.
    """

    def __init__(
        self,
        embedding_model,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "semantic_cache_index",
    ):
        """Initializes the semantic cache handler.

        Args:
            embedding_model: LangChain embedding model.
            redis_url: Redis server URL.
            index_name: Name of the semantic cache index.
        """
        self.embedding_model = embedding_model
        self.redis_url = redis_url
        self.index_name = index_name
        self.redis_client = get_redis_client(redis_url)
        self.vector_index = VectorSearchIndex(
            redis_client=self.redis_client,
            index_name=index_name,
            vector_dimension=1536,
            distance_metric="COSINE",
        )

    def save_qa_pair(
        self, question: str, answer: str, metadata: dict = None
    ) -> bool:
        """Stores a question-answer pair in the semantic cache.

        Args:
            question: The question text (used for embedding).
            answer: The answer text.
            metadata: Optional additional metadata.

        Returns:
            True if the pair was saved successfully.
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
                doc_id=key, embedding=embedding, metadata=doc_metadata
            )
        except Exception as e:
            print(f"[SemanticCache] Save error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def search_similar_question(
        self, query: str, top_k: int = 3, score_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Finds cached answers for semantically similar questions.

        Args:
            query: The question to search for.
            top_k: Maximum number of results.
            score_threshold: Minimum cosine similarity threshold.

        Returns:
            List of dicts with question, answer, and similarity fields.
        """
        try:
            embedding = self.embedding_model.embed_query(query)
            results = self.vector_index.search_similar(
                query_vector=embedding, top_k=top_k, score_threshold=score_threshold
            )
            return [
                {
                    "question": r["metadata"].get("question"),
                    "answer": r["metadata"].get("answer"),
                    "similarity": r["similarity"],
                }
                for r in results
            ]
        except Exception as e:
            print(f"[SemanticCache] Search error: {e}")
            import traceback

            traceback.print_exc()
            return []


class EmbeddingsCacheHandler:
    """Caches raw embedding vectors in Redis keyed by text SHA-256 hash.

    Attributes:
        prefix: Redis key prefix for cached embeddings.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "embeddings_cache",
    ):
        """Initializes the embeddings cache handler.

        Args:
            redis_url: Redis server URL.
            prefix: Redis key prefix.
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.redis_client = get_redis_client(redis_url)

    def _make_key(self, text: str) -> str:
        """Generates a Redis key from the SHA-256 hash of the text."""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.prefix}:{h}"

    def get_embedding(self, text: str) -> np.ndarray | None:
        """Retrieves a cached embedding for the given text.

        Args:
            text: The original text.

        Returns:
            Cached embedding as a numpy array, or None if not found.
        """
        key = self._make_key(text)
        value = self.redis_client.get(key)
        if value is not None:
            return np.frombuffer(value, dtype=np.float32)
        return None

    def set_embedding(self, text: str, embedding: np.ndarray):
        """Stores an embedding in the cache.

        Args:
            text: The original text (used to generate the cache key).
            embedding: Embedding vector to cache.
        """
        key = self._make_key(text)
        self.redis_client.set(key, embedding.astype(np.float32).tobytes())
