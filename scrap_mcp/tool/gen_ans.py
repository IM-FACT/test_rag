from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def load_prompt(filename):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_dir = os.path.join(root_dir, "prompts")
    filepath = os.path.join(prompts_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# GPT-4.1을 사용하여 MCP 수집 문서 기반 질문 답변 생성
def ans_with_mcp(query: str, docs: List[Dict[str, str]]) -> str:
    try:
        action_prompt = load_prompt("generate_ans_prompt.txt")
        pool = "\n\n".join(f"[출처] {doc['url']}\n{doc['content']}" for doc in docs)


        user_prompt = f"""{action_prompt}
                        [문서 정보] {pool}
                        [질문] {query} """

        response = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": action_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT 4.1 답변 생성 실패")
        print(e)
        return "답변 생성 실패"