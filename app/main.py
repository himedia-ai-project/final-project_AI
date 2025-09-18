import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel, Field
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
    pdf_id: int = Field(alias="productId")
    question: str
    chatMessage: List[Dict[str, str]] = Field(default_factory=list, alias="messages")

@app.post("/chat")
async def chat(request: ChatIn):
    history: List[BaseMessage] = []

    # chatMessage → BaseMessage 변환 (여기서 바로 처리)
    if request.chatMessage:
        for m in request.chatMessage:
            role = m.get("role")
            messages = m.get("messages")
            if role == "user":
                history.append(HumanMessage(content=messages))
            elif role == "bot":
                history.append(AIMessage(content=messages))

    state: QueryState = {
        "pdf_id": request.pdf_id,
        "question": request.question,
        "history": history,
    }

    # 그래프 실행
    result = await query_graph.ainvoke(state)

    return result["answer"]
