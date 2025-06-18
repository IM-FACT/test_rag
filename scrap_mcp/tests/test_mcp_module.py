# test_mcp_module.py
import asyncio
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from scrap_mcp.mcp_module import search_scrap
from scrap_mcp.tool.gen_ans import ans_with_mcp

query = "파리 올림픽 보다가 알게 됐는데, 파리가 탄소중립 도시라고 해서 에어컨도 안 틀었다고 들음. 탄소중립 도시가 된 배경이 궁금함"

docs = asyncio.run(search_scrap(query))
print(f"\n수집된 문서 개수: {len(docs)}개")
for i, doc in enumerate(docs, 1):
    print(f"\n- {i}번 문서")
    print(doc[:1000])  # 너무 길면 잘라서 출력

# GPT 4.1로 답변 생성
print("\nGPT 4.1 답변 생성 중...\n")
answer = ans_with_mcp(query, docs)

# 결과 출력
print("최종 답변:\n")
print(answer)