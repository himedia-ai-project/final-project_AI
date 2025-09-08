from typing import List
from typing_extensions import NotRequired, TypedDict
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    file_path: str
    pdf_id: str
    documents: NotRequired[List]
    chunks: NotRequired[List]
    vectorstore: NotRequired[FAISS]
    store_path: NotRequired[str]


class QueryState(TypedDict):
    pdf_id: str
    question: str
    vectorstore: NotRequired[FAISS]
    context: NotRequired[str]
    answer: NotRequired[str]
    history: NotRequired[List[BaseMessage]]
