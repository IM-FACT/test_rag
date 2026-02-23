"""Main RAG processor orchestrating search, caching, and answer generation.

Coordinates the full RAG pipeline: semantic cache lookup, vector search
for relevant documents, MCP-based web scraping fallback, and GPT answer
generation with automatic cache storage.
"""

import sys
from typing import Dict, Any
import time
import asyncio

from langchain.embedding_generator import EmbeddingGenerator
from langchain.redis_handler import RedisVectorSearchHandler, SemanticCacheHandler
from scrap_mcp.mcp_module import search_scrap
from scrap_mcp.tool.gen_ans import ans_with_mcp

SIMILARITY_THRESHOLD = 0.4


class MainProcessor:
    """Orchestrates the LangChain-based RAG pipeline.

    Attributes:
        embedding_generator: Generates text embeddings.
        redis_handler: Handles vector search operations.
        semantic_cache: Manages semantic question-answer caching.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initializes the main processor.

        Args:
            redis_url: Redis server URL.
        """
        try:
            self.embedding_generator = EmbeddingGenerator()
            self.redis_handler = RedisVectorSearchHandler(
                embedding_model=self.embedding_generator.embeddings,
                redis_url=redis_url,
                index_name="document_index",
            )
            self.semantic_cache = SemanticCacheHandler(
                embedding_model=self.embedding_generator.embeddings,
                redis_url=redis_url,
            )
            print("MainProcessor initialized.")
        except Exception as e:
            print(f"MainProcessor initialization error: {e}")
            sys.exit(1)

    def process(self, query: str) -> Dict[str, Any]:
        """Processes a user query through the full RAG pipeline.

        Pipeline steps:
            1. Semantic cache lookup for previously answered similar questions.
            2. Vector search for relevant stored documents.
            3. MCP web scraping fallback if no documents are found.
            4. GPT answer generation from collected documents.
            5. Cache the new question-answer pair.

        Args:
            query: User's question.

        Returns:
            Dict containing operation status, answer, and search results.
        """
        result = {
            "success": False,
            "operation": None,
            "message": "",
            "similar_items": [],
            "cache_answer": None,
            "vector_search_results": [],
            "final_answer": None,
        }

        # Step 1: Semantic cache lookup
        cache_results = self.semantic_cache.search_similar_question(
            query=query, score_threshold=0.05
        )
        if cache_results:
            best = max(cache_results, key=lambda x: x["similarity"])
            result["operation"] = "cache_hit"
            result["cache_answer"] = best["answer"]
            result["success"] = True
            result["message"] = "Answer returned from semantic cache."
            result["final_answer"] = best["answer"]
            return result

        # Step 2: Vector search for stored documents
        vector_results = self.redis_handler.search_similar_embeddings(
            query_text=query, top_k=3, similarity_threshold=0.4
        )
        result["vector_search_results"] = vector_results

        if vector_results:
            print("[Vector DB HIT] Generating answer from similar documents.")
            query_ans_pool = [item["metadata"]["text"] for item in vector_results]
        else:
            print("MCP search started...")
            query_ans_pool = asyncio.run(search_scrap(query))
            print(f"MCP document collection complete: {len(query_ans_pool)} docs")

        # Step 3: GPT answer generation
        print(f"GPT answer generation started ({len(query_ans_pool)} docs)...")
        generated_answer = ans_with_mcp(query=query, docs=query_ans_pool)
        print(f"GPT answer generation complete (length: {len(generated_answer)})")

        # Step 4: Store in semantic cache
        self.semantic_cache.save_qa_pair(
            question=query,
            answer=generated_answer,
            metadata={"source": "gpt", "timestamp": time.time()},
        )
        result["operation"] = "cache_miss_saved"
        result["cache_answer"] = generated_answer
        result["success"] = True
        result["message"] = "New answer generated and saved to semantic cache."
        result["final_answer"] = generated_answer
        return result

    def display_results(self, result: Dict[str, Any]) -> None:
        """Prints processing results to the console.

        Args:
            result: Processing result dict from process().
        """
        if not result["success"]:
            print(f"\nError: {result['message']}")
            return

        if result["operation"] == "cache_hit":
            print("\n[Semantic Cache HIT] Answer:")
            print("-" * 80)
            print(result["cache_answer"])
            print("-" * 80)
        elif result["operation"] == "cache_miss_saved":
            print("\n[Semantic Cache MISS] New answer saved:")
            print("-" * 80)
            print(result["cache_answer"])
            print("-" * 80)
            print("\n[Vector Search Results]")
            for idx, item in enumerate(result["vector_search_results"], 1):
                text = item["metadata"].get("text", "")
                similarity = item["similarity"]
                print(f"{idx}. {text} (similarity: {similarity:.2f})")
            print("-" * 80)


def main():
    """Entry point for the RAG processor."""
    user_query = (
        "해수면 상승으로 실제 우리나라 해안가 도시들이 위험할까? "
        "몇 년 뒤에 어떤 변화가 생길지 궁금함"
    )

    processor = MainProcessor()

    print("\n=== RAG Processor ===")
    print(f"Query: {user_query}")
    print("=" * 80)

    result = processor.process(query=user_query)
    processor.display_results(result)


if __name__ == "__main__":
    main()
