import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from app.rag.state import GraphState, QueryState
from app.rag.workflow import pdf_graph, query_graph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

app = FastAPI(title="PDF RAG API")
# session 저장소
session_store: Dict[str, dict[str, Any]] = {}


@app.post("/upload")
async def file_upload(file: UploadFile = File(...),
                      pdf_id: int = Form(...)):

    # 1. 업로드 파일 저장
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    initial_state: GraphState = {"file_path": temp_path, "pdf_id": pdf_id}
    final_state = await pdf_graph.ainvoke(initial_state)

    # 2. 임시파일 삭제
    os.remove(temp_path)

    return {
        "message": f"PDF {file.filename} uploaded successfully",
        "pdf_id": final_state.get("pdf_id"),
        "store_path": final_state.get("store_path"),
    }


class ChatIn(BaseModel):
    pdf_id: int
    question: str
    history: list[dict[str, str]] | None = None

@app.post("/chat")
async def chat(in_: ChatIn):
    messages: List[BaseMessage] = []

    # history → BaseMessage 변환 (여기서 바로 처리)
    if in_.history:
        for m in in_.history:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != in_.question:
        messages.append(HumanMessage(content=in_.question))

    state: QueryState = {
        "pdf_id": in_.pdf_id,
        "question": in_.question,
        "history": messages,
    }

    # 그래프 실행
    result = await query_graph.ainvoke(state)

    return result["answer"]
