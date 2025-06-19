import requests
from typing import List, Dict
import os

def brave_search_impl(query: str, api_key: str, count: int=3) -> List[Dict[str, str]]:
    url="https://api.search.brave.com/res/v1/web/search"
    headers={
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    allowed_sites=["ipcc.ch",
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
    "mbc.co.kr"]

    # "nature.com",
    # "sciencemag.org",
    # "kbs.co.kr",
    # "mbc.co.kr",
    # "sbs.co.kr",
    # "yna.co.kr",
    # "ytn.co.kr"


    if allowed_sites:
        site_filter = " OR ".join([f"site:{site}" for site in allowed_sites])
        query = f"{query} {site_filter}"

    params={
        "q": query,
        "count": count
    }

    response = requests.get(url, headers=headers, params=params)
    results = response.json().get("web", {}).get("results", [])
    return [
        {
            "title":res.get("title",""),
            "text": res.get("description", ""),
            "url": res.get("url", "")
        }
        for res in results
    ]