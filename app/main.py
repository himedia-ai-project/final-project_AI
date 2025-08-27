import os
from fastapi import FastAPI, File, Query, UploadFile
from app.rag.state import GraphState, QueryState
from app.rag.workflow import pdf_graph, query_graph
from langchain_community.vectorstores import FAISS

app = FastAPI(title="PDF RAG API")
vectorstore: FAISS | None = None


@app.post("/upload")
async def file_upload(file: UploadFile = File(...)):

    # 1. 업로드 파일 저장
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    initial_state: GraphState = {"file_path": temp_path}
    final_state = await pdf_graph.ainvoke(initial_state)

    # 2. 임시파일 삭제
    os.remove(temp_path)

    return {
        "message": f"PDF {file.filename} uploaded successfully",
        "pdf_id": final_state.get("pdf_id"),
        "store_path": final_state.get("store_path"),
    }


@app.get("/query")
async def query_rag(question: str = Query(..., description="질문할 내용을 입력")):
    if not hasattr(app.state, "vectorstore"):
        return {"error": "먼저 PDF를 업로드하고 벡터스토어를 생성하세요."}

    state: QueryState = {"question": question, "vectorstore": app.state.vectorstore}

    # 3️⃣ query workflow 실행
    final_state = query_graph.invoke(state)

    return {
        "query": question,
        "answer": final_state.get("answer", ""),
        "context": final_state.get("context", ""),
    }
