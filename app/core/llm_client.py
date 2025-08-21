import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()  # 필요시 경로 지정 가능

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

# 테스트 출력
