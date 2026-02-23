"""Google Custom Search API module.

Searches Google Custom Search for a URL to retrieve its title and
description metadata.
"""

from dotenv import load_dotenv
import aiohttp
import asyncio
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")


async def use_google(url: str) -> dict | str:
    """Retrieves title and description for a URL via Google Custom Search.

    Args:
        url: Target URL to search for.

    Returns:
        Dict with 'title' and 'description' fields on success,
        "Fail" if no matching result is found,
        or the error message string on exception.
    """
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CX,
                "q": url,
                "num": 8,
            }
            async with session.get(
                "https://www.googleapis.com/customsearch/v1", params=params
            ) as resp:
                data = await resp.json()

        response = {}
        for q in data["items"]:
            if q["link"] == url:
                response["title"] = q["title"]
                try:
                    response["description"] = q["pagemap"]["metatags"][0][
                        "og:description"
                    ]
                except (KeyError, IndexError):
                    try:
                        response["description"] = q["snippet"]
                    except KeyError:
                        response["description"] = (
                            "No description is available for this content"
                        )

        if not response:
            response = "Fail"
        return response
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print(
        asyncio.run(
            use_google(
                "https://www.me.go.kr/home/file/readDownloadFile.do"
                "?fileId=97828&fileSeq=1&openYn=Y"
            )
        )
    )
