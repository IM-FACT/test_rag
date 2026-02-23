"""Direct HTTP text extraction using trafilatura."""

from trafilatura import extract
import requests


def use_trafilatura(url: str) -> str:
    """Extracts main text content from a URL via HTTP GET and trafilatura.

    Args:
        url: Target URL to extract text from.

    Returns:
        Extracted text content, "Fail" if extraction yields nothing,
        or the error message string on exception.
    """
    try:
        response = requests.get(url)
        text = extract(response.text)
        if text is None:
            return "Fail"
        return text
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print(use_trafilatura("https://www.kma.go.kr/kids/231.jsp"))
