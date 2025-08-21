from typing import TypedDict
from langchain.docstore.document import Document


class GraphState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
