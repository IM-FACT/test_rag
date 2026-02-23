"""Brave Search API implementation.

Searches the Brave Search API with optional site filtering to retrieve
web results from trusted sources.
"""

import requests
from typing import List, Dict


# Trusted domains for climate-related content.
ALLOWED_SITES = [
    "ipcc.ch",
    "nasa.gov",
    "kma.go.kr",
    "me.go.kr",
    "unep.org",
    "noaa.gov",
    "berkeleyearth.org",
    "nature.com",
    "sciencemag.org",
    "news.kbs.co.kr",
    "news.sbs.co.kr",
    "news.ytn.co.kr",
    "yna.co.kr",
    "mbc.co.kr",
]


def brave_search_impl(
    query: str, api_key: str, count: int = 3
) -> List[Dict[str, str]]:
    """Searches the Brave Search API with site filtering.

    Args:
        query: Search query string.
        api_key: Brave Search API subscription token.
        count: Number of results to retrieve.

    Returns:
        List of result dicts with 'title', 'text', and 'url' fields.
    """
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }

    if ALLOWED_SITES:
        site_filter = " OR ".join(f"site:{site}" for site in ALLOWED_SITES)
        query = f"{query} {site_filter}"

    params = {"q": query, "count": count}

    response = requests.get(url, headers=headers, params=params)
    results = response.json().get("web", {}).get("results", [])
    return [
        {
            "title": res.get("title", ""),
            "text": res.get("description", ""),
            "url": res.get("url", ""),
        }
        for res in results
    ]
