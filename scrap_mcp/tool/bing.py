from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from trafilatura import extract
import asyncio
import lxml
import time

async def use_bing_n_page(url: str):
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

            soup = BeautifulSoup(text, 'lxml')

            b_result = soup.find("ol", id = "b_results")
            elems = b_result.find_all("li", class_ = "b_algo")[:8]
            #elems.append(b_result.find("li", class_ = "b_algo b_vtl_deeplinks"))
            for elem in elems:
                title_n_link = elem.find("h2").find("a")
                caption = elem.find("div", class_ = "b_caption")
                caption2 = elem.find("p", class_ = "b_lineclamp3")
                caption3 = elem.find("p", class_ = "b_lineclamp2")

                title = title_n_link.get_text(strip=True) if title_n_link else None
                link = title_n_link['href'] if title_n_link else None

                if caption:
                    descrption = caption.get_text(strip=True)
                elif caption2:
                    descrption = caption2.get_text(strip=True)
                elif caption3:
                    descrption = caption3.get_text(strip=True)
                else:
                    descrption = None

                if(link == url):
                    result["title"] = title
                    result["descrption"] = descrption
                    break

            if not result:
                result["Bing"] = "Fail : No result"
            content = await page2.content()
            result["content"] = extract(content)
                
        except Exception as e:
            result = "해당 링크가 파일 다운로드이거나, 런타임에 오류가 발생했습니다."
            result += f"Error: {e}"
        finally: 
            await browser.close()
    return result


if __name__ == "__main__":
    print(asyncio.run(use_bing_n_page("https://news.kbs.co.kr/news/pc/view/view.do?ncd=5528082")))