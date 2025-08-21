from fastapi import FastAPI, Query
from app.rag.graph_builder import run_rag_graph

app = FastAPI(title="PDF RAG API")


@app.get("/ask")
def ask_question(question: str = Query(..., description="질문 입력")):
    answer = run_rag_graph(question)
    return {"question": question, "answer": answer}
