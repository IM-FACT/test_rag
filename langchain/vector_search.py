"""Vector search module using Redis 8 and HNSW algorithm.

Provides vector indexing and similarity search capabilities backed by
Redis Vector Search with the HNSW (Hierarchical Navigable Small World)
algorithm.
"""

import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from typing import List, Dict, Any


class VectorSearchIndex:
    """Manages a Redis Vector Search index for similarity queries.

    Attributes:
        redis_client: Redis client instance.
        index_name: Name of the search index.
        vector_dimension: Dimensionality of stored vectors.
        distance_metric: Distance metric (COSINE, L2, or IP).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        index_name: str = "climate_vectors",
        vector_dimension: int = 1536,
        distance_metric: str = "COSINE",
    ):
        """Initializes the vector search index.

        Args:
            redis_client: Redis client instance.
            index_name: Name of the search index.
            vector_dimension: Vector dimensionality (1536 for
                OpenAI text-embedding-3-small).
            distance_metric: Distance metric (COSINE, L2, or IP).
        """
        self.redis_client = redis_client
        self.index_name = index_name
        self.vector_dimension = vector_dimension
        self.distance_metric = distance_metric

        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Creates the index if it does not already exist."""
        try:
            self.redis_client.ft(self.index_name).info()
            print(f"Index '{self.index_name}' already exists.")
        except Exception:
            print(f"Creating index '{self.index_name}'...")
            self._create_index()

    def _create_index(self):
        """Creates a new HNSW vector search index in Redis."""
        try:
            schema = (
                VectorField(
                    "embedding_vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dimension,
                        "DISTANCE_METRIC": self.distance_metric,
                        "INITIAL_CAP": 10000,
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
                TextField("question", sortable=True),
                TextField("source_url"),
                TextField("text"),
                NumericField("timestamp", sortable=True),
                TextField("custom_key"),
                TextField("id"),
            )

            definition = IndexDefinition(
                prefix=[f"doc:{self.index_name}:"], index_type=IndexType.HASH
            )

            self.redis_client.ft(self.index_name).create_index(
                fields=schema, definition=definition
            )
            print(f"Index '{self.index_name}' created.")
        except Exception as e:
            print(f"Index creation error: {e}")
            raise

    def add_document(
        self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> bool:
        """Adds a document with its embedding vector to the index.

        Args:
            doc_id: Unique document identifier.
            embedding: Embedding vector.
            metadata: Document metadata fields.

        Returns:
            True if the document was stored successfully.
        """
        try:
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

            doc_data = metadata.copy()
            doc_data["embedding_vector"] = embedding_bytes
            doc_data["custom_key"] = doc_id
            doc_data["id"] = doc_id

            redis_key = f"doc:{self.index_name}:{doc_id}"
            self.redis_client.hset(redis_key, mapping=doc_data)
            return True
        except Exception as e:
            print(f"Document add error: {e}")
            return False

    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Searches for similar vectors using the HNSW index.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: Minimum cosine similarity threshold.

        Returns:
            List of result dicts containing key, similarity, and metadata.
        """
        try:
            query_bytes = np.array(query_vector, dtype=np.float32).tobytes()

            base_query = f"*=>[KNN {top_k} @embedding_vector $vector AS score]"
            query = (
                Query(base_query).sort_by("score").paging(0, top_k).dialect(2)
            )

            results = self.redis_client.ft(self.index_name).search(
                query, query_params={"vector": query_bytes}
            )

            formatted_results = []
            for doc in results.docs:
                similarity_score = 1 - float(doc.score)

                if similarity_score >= score_threshold:
                    metadata_from_doc = {
                        "question": getattr(doc, "question", None),
                        "source_url": getattr(doc, "source_url", None),
                        "text": getattr(doc, "text", None),
                        "timestamp": (
                            float(getattr(doc, "timestamp", 0))
                            if hasattr(doc, "timestamp")
                            else None
                        ),
                        "id": getattr(doc, "id", None),
                        "custom_key": getattr(doc, "custom_key", None),
                        "answer": getattr(doc, "answer", None),
                        "type": getattr(doc, "type", None),
                    }

                    result_item = {
                        "key": getattr(doc, "id", None),
                        "similarity": similarity_score,
                        "metadata": metadata_from_doc,
                        "redis_key": getattr(doc, "id", None),
                        "question": getattr(doc, "question", None),
                        "answer": getattr(doc, "answer", None),
                    }
                    formatted_results.append(result_item)

            return formatted_results
        except Exception as e:
            print(f"Vector search error: {e}")
            import traceback

            traceback.print_exc()
            return []

    def update_ef_runtime(self, ef_runtime: int = 10):
        """Adjusts the HNSW runtime search parameter.

        Args:
            ef_runtime: Number of neighbors to explore during search.
                Higher values improve accuracy but reduce speed.
        """
        try:
            self.redis_client.ft(self.index_name).config_set(
                "HNSW_EF_RUNTIME", ef_runtime
            )
            print(f"EF_RUNTIME set to {ef_runtime}.")
        except Exception as e:
            print(f"EF_RUNTIME configuration error: {e}")

    def get_index_info(self) -> Dict[str, Any]:
        """Returns metadata about the current index."""
        try:
            return self.redis_client.ft(self.index_name).info()
        except Exception as e:
            print(f"Index info retrieval error: {e}")
            return {}

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from the index.

        Args:
            doc_id: Identifier of the document to delete.

        Returns:
            True if the document was deleted successfully.
        """
        try:
            redis_key = f"doc:{self.index_name}:{doc_id}"
            result = self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            print(f"Document deletion error: {e}")
            return False
