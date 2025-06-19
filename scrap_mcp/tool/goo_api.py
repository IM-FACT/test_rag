from dotenv import load_dotenv
import aiohttp
import asyncio
import os
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

async def use_google(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "key": GOOGLE_API_KEY,
                "cx": "f14293bc90ca74970",
                "q": url,
                "num": 8,
            }
            async with session.get("https://www.googleapis.com/customsearch/v1", params=params) as response:
                data = await response.json()
        response : str = {}
        for q in data["items"]:
            if q["link"] == url:
                response["title"] = q["title"]
                try:
                    response["description"] = q["pagemap"]["metatags"][0]["og:description"]
                except:
                    try:
                        response["description"] = q["snippet"]
                    except:
                        response["description"] = "No description is available for this content"

        if not response:
            response = "Fail"
        return response
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    print(asyncio.run(use_google("https://www.me.go.kr/home/file/readDownloadFile.do?fileId=97828&fileSeq=1&openYn=Y")))