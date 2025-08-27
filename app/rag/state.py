from typing import Annotated, List
from numpy import long, number
from typing_extensions import NotRequired, TypedDict
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    file_path: str
    pdf_id: NotRequired[str]
    documents: NotRequired[List]
    chunks: NotRequired[List]
    vectorstore: NotRequired[FAISS]
    store_path: NotRequired[str]


class QueryState(TypedDict):
    question: str
    vectorstore: FAISS
    context: NotRequired[str]
    answer: NotRequired[str]
