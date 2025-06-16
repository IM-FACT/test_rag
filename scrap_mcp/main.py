from mcp.server.fastmcp import FastMCP
from typing import Union
import json
import asyncio
import sys
import io
import re

from scrap_mcp.tool.text import use_tra
from scrap_mcp.tool.bing import use_bing_n_page
from scrap_mcp.tool.goo_api import use_google

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
mcp = FastMCP("Scraper")

@mcp.tool()
async def scrape_web(url: str, keyword: Union[str, list[str]] = "default") -> str:
    """
    Scrape a website of given URL and extract context of given keyword and return all text in JSON style.
    URL of contents is in "url" field.
    Google search results on URL are in "google" field.
    General HTTP Request result is in "normal" field.
    Using web browser to get the content of the page is in "page" field. 
    In page field, title and description field is on Bing search result.
    Content field is extracted text from the page.
    """

    result = {}
    result["url"] = url
    tasks = [use_bing_n_page(url), use_google(url), asyncio.to_thread(use_tra, url)]
    results = await asyncio.gather(*tasks)

    result["page"] = results[0]
    result["google"] = results[1]
    result["normal"] = results[2]

    
    response = json.dumps(result, ensure_ascii=False)
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)
    
    return cleaned



if __name__ == '__main__':
    #asyncio.run(scrape_web("https://www.bbc.com/news/articles/cd0n1m4r99zo"))
    asyncio.run(mcp.run())
    # texts = load_page('https://edition.cnn.com/2025/05/19/middleeast/socotra-trees-yemen-climate-change-intl-hnk')
    # contexts = keyword_context(texts, 'blood', context_range=150)
    
    
    # for i in range(len(contexts)):
    #     print(contexts[i])
    #     print('-'*100)