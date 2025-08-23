from typing import List
from typing_extensions import NotRequired, TypedDict
from langchain_community.vectorstores import FAISS


class GraphState(TypedDict):
    file_path: str
    documents: NotRequired[List]
    chunks: NotRequired[List]
    vectorstore: NotRequired[FAISS]


class QueryState(TypedDict):
    question: str
    vectorstore: FAISS
    context: NotRequired[str]
    answer: NotRequired[str]
