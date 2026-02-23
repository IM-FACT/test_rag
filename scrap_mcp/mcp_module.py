"""MCP module for query rewriting, web search, and content scraping.

Orchestrates the search pipeline: rewrites the user query into multiple
search variants, runs Brave Search for relevant URLs, validates them,
and scrapes content with priority-based extraction.
"""

import sys
import os
import requests
import asyncio
import json

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from scrap_mcp.brave_search_module.brave_search_impl import brave_search_impl
from scrap_mcp.tool.rewrite_query import rewrite_query
from scrap_mcp.main import scrape_web

load_dotenv()
api_key = os.getenv("BRAVE_AI_API_KEY")


def is_url_alive(url: str) -> bool:
    """Checks whether a URL is reachable and returns valid content.

    Args:
        url: URL to validate.

    Returns:
        True if the URL responds with status 200 and no error indicators.
    """
    try:
        if "jsessionid" in url.lower():
            return False
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code != 200:
            return False
        error_keywords = ["존재하지 않는", "not found", "세션이 만료"]
        return not any(keyword in res.text.lower() for keyword in error_keywords)
    except requests.RequestException:
        return False


async def search_scrap(query: str) -> list[dict]:
    """Searches the web and scrapes content for the given query.

    Pipeline:
        1. Rewrites the query into Korean and English variants.
        2. Runs Brave Search for each variant.
        3. Filters out unreachable URLs.
        4. Scrapes each valid URL and extracts content.
        5. Returns the top 3 documents (max 5000 chars each).

    Args:
        query: User's original question.

    Returns:
        List of dicts with 'url' and 'content' fields.
    """
    kor_queries, eng_queries = rewrite_query(query)
    rewritten_query_list = kor_queries + eng_queries
    print(f"\nRewritten queries: {rewritten_query_list}")

    results = []
    for r_q in rewritten_query_list:
        partial_results = brave_search_impl(query=r_q, api_key=api_key, count=2)
        results.extend(partial_results)

    valid_results = [res for res in results if is_url_alive(res["url"])]

    all_scrap_list = []
    for item in valid_results:
        try:
            result_json = await scrape_web(item["url"], rewritten_query_list)
            result = json.loads(result_json)

            # Content priority: normal > google > page.description
            content = result.get("normal")
            if not content or not isinstance(content, str) or len(content.strip()) == 0:
                content = result.get("google")
            if not content or not isinstance(content, str) or len(content.strip()) == 0:
                content = result.get("page", {}).get("description", "")

            if content and isinstance(content, str) and len(content.strip()) > 0:
                all_scrap_list.append(
                    {"url": item["url"], "content": content.strip()}
                )
        except Exception as e:
            print(f"Scrape failed for {item['url']}: {e}")

    docs = all_scrap_list[:3]
    for doc in docs:
        doc["content"] = doc["content"][:5000]
    return docs
