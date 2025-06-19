from trafilatura import extract
import requests

def use_tra(url: str):
    try:
        qht = requests.get(url)
        text = extract(qht.text)
        if text is None:
            return "Fail"
        return text

    except Exception as e:
        return str(e)
    

if __name__ == "__main__":
    print(use_tra("https://www.kma.go.kr/kids/231.jsp"))