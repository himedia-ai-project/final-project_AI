import os
from typing import Any, Dict, cast
from fastapi import FastAPI, File, UploadFile, Form
from app.rag.state import GraphState, QueryState
from app.rag.workflow import pdf_graph, query_graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="PDF RAG API")
# session 저장소
session_store: Dict[str, dict[str, Any]] = {}


@app.post("/upload")
async def file_upload(file: UploadFile = File(...),
                      pdf_id: str = Form(...)):

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


@app.post("/chat")
async def chat(user_id: int, pdf_id: str, question: str):
    # 1. 세션 키 생성
    session_key = f"{user_id}:{pdf_id}"

    # 2. 기존 세션 불러오기 or 새로 생성
    if session_key in session_store:
        state: QueryState = cast(QueryState, session_store[session_key])
    else:
        state: QueryState = {
            "pdf_id": pdf_id,
            "question": question,
            "history": [],
        }

    # 3. 중복 질문 방지
    messages = state.get("history", [])
    if (
        not messages
        or not isinstance(messages[-1], HumanMessage)
        or messages[-1].content != question
    ):
        messages.append(HumanMessage(content=question))
    state["history"] = messages

    # 4. 그래프 실행
    result = await query_graph.ainvoke(state)

    # 최신 state 저장
    session_store[session_key] = result

    return {
        "pdf_id": pdf_id,
        "question": question,
        "answer": result["answer"],
        "history": [
            {"role": m.type, "content": m.content} for m in result.get("history", [])
        ],
    }
