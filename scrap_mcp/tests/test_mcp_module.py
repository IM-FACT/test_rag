"""Integration test for the MCP search and answer generation pipeline."""

import asyncio
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from scrap_mcp.mcp_module import search_scrap
from scrap_mcp.tool.gen_ans import ans_with_mcp

query = (
    "23~24년 미국에 큰 허리케인이 연달아서 왔다는 뉴스를 본 것 같은데 "
    "이것도 이상기후 때문인가? 원래 이랬던 건지 아니면 더 심해진건지"
)

docs = asyncio.run(search_scrap(query))
for doc in docs:
    doc["content"] = doc["content"][:5000]

print(f"\nCollected documents: {len(docs)}")
for i, doc in enumerate(docs, 1):
    print(f"\n- Document {i}")
    print(doc["content"][:1000])

print("\nGenerating GPT answer...\n")
answer = ans_with_mcp(query, docs)

print("Final answer:\n")
print(answer)
