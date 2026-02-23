"""Query rewriting module using GPT-4o.

Rewrites a user query into multiple Korean and English search variants
to improve search coverage across different search engines.
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
import json

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


def rewrite_query(user_query: str) -> tuple[list[str], list[str]]:
    """Rewrites a user query into Korean and English search variants.

    Uses GPT-4o to generate multiple search-optimized reformulations
    of the original query in both languages.

    Args:
        user_query: Original user question.

    Returns:
        Tuple of (korean_queries, english_queries) lists.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": load_prompt("rewrite_query_prompt.txt"),
                },
                {"role": "user", "content": user_query},
            ],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return result["korean"], result["english"]
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return [user_query], []
