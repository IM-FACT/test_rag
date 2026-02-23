"""Bing search and page content extraction using Playwright.

Searches Bing for a given URL to retrieve its title and description,
while simultaneously extracting the page's full text content via
trafilatura.
"""

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from trafilatura import extract
import asyncio
import lxml


async def use_bing_and_page(url: str) -> dict | str:
    """Retrieves Bing search metadata and page content for a URL.

    Opens two browser tabs in parallel: one searches Bing for the URL
    to get its title/description, and the other loads the page directly
    for text extraction.

    Args:
        url: Target URL to search and extract.

    Returns:
        Dict with 'title', 'description', and 'content' fields on success,
        or an error message string on failure.
    """
    result = {}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            page2 = await browser.new_page()
            await page2.goto(url)

            await page.goto("https://www.bing.com")
            await page.get_by_role("search").type(url)
            await page.get_by_role("search").press("Enter")
            waiting = page.locator("h2").first
            await waiting.wait_for(state="visible")
            text = await page.content()

            soup = BeautifulSoup(text, "lxml")
            b_result = soup.find("ol", id="b_results")
            elems = b_result.find_all("li", class_="b_algo")[:8]

            for elem in elems:
                title_n_link = elem.find("h2").find("a")
                caption = elem.find("div", class_="b_caption")
                caption2 = elem.find("p", class_="b_lineclamp3")
                caption3 = elem.find("p", class_="b_lineclamp2")

                title = title_n_link.get_text(strip=True) if title_n_link else None
                link = title_n_link["href"] if title_n_link else None

                if caption:
                    description = caption.get_text(strip=True)
                elif caption2:
                    description = caption2.get_text(strip=True)
                elif caption3:
                    description = caption3.get_text(strip=True)
                else:
                    description = None

                if link == url:
                    result["title"] = title
                    result["description"] = description
                    break

            if not result:
                result["Bing"] = "Fail: No result"

            content = await page2.content()
            result["content"] = extract(content)
        except Exception as e:
            result = f"Page is a file download or a runtime error occurred. Error: {e}"
        finally:
            await browser.close()
    return result


if __name__ == "__main__":
    print(
        asyncio.run(
            use_bing_and_page(
                "https://news.kbs.co.kr/news/pc/view/view.do?ncd=5528082"
            )
        )
    )
