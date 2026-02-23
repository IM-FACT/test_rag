"""Answer generation module using GPT-4 Turbo.

Generates answers based on collected documents using a prompt template
and the OpenAI GPT-4 Turbo model.
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_prompt(filename: str) -> str:
    """Loads a prompt template from the prompts directory.

    Args:
        filename: Name of the prompt file.

    Returns:
        Prompt text content.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_dir = os.path.join(root_dir, "prompts")
    filepath = os.path.join(prompts_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def ans_with_mcp(query: str, docs: List[Dict[str, str]]) -> str:
    """Generates an answer from collected documents using GPT-4 Turbo.

    Args:
        query: User's original question.
        docs: List of document dicts with 'url' and 'content' fields.

    Returns:
        Generated answer text, or a failure message on error.
    """
    try:
        action_prompt = load_prompt("generate_ans_prompt.txt")
        pool = "\n\n".join(
            f"[출처] {doc['url']}\n{doc['content']}" for doc in docs
        )

        user_prompt = (
            f"{action_prompt}\n"
            f"[문서 정보] {pool}\n"
            f"[질문] {query}"
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": action_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT answer generation failed: {e}")
        return "Answer generation failed."
