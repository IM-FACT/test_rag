"""MCP server for web scraping.

Exposes a scrape_web tool via the Model Context Protocol that collects
page content from multiple sources (Bing, Google Custom Search, and
direct HTTP extraction) in parallel.
"""

from mcp.server.fastmcp import FastMCP
from typing import Union
import json
import asyncio
import sys
import io
import re

from scrap_mcp.tool.text import use_trafilatura
from scrap_mcp.tool.bing import use_bing_and_page
from scrap_mcp.tool.goo_api import use_google

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")
mcp = FastMCP("Scraper")


@mcp.tool()
async def scrape_web(
    url: str, keyword: Union[str, list[str]] = "default"
) -> str:
    """Scrapes a website and extracts content from multiple sources.

    Fetches page content in parallel from three sources:
    - Bing search results and direct page extraction via Playwright.
    - Google Custom Search API metadata.
    - Direct HTTP request with trafilatura extraction.

    Args:
        url: Target URL to scrape.
        keyword: Search keyword(s) for context filtering.

    Returns:
        JSON string with url, page, google, and normal fields.
    """
    result = {"url": url}
    tasks = [
        use_bing_and_page(url),
        use_google(url),
        asyncio.to_thread(use_trafilatura, url),
    ]
    results = await asyncio.gather(*tasks)

    result["page"] = results[0]
    result["google"] = results[1]
    result["normal"] = results[2]

    response = json.dumps(result, ensure_ascii=False)
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", response)
    return cleaned


if __name__ == "__main__":
    asyncio.run(mcp.run())
