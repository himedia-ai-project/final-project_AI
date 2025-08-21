from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from app.rag.nodes import generate_node_fn, search_node_fn
from app.rag.state import GraphState


# --- StateGraph 빌더 ---
graph_builder = StateGraph(GraphState)

# 노드 추가
graph_builder.add_node("search_node", search_node_fn)
graph_builder.add_node("generate_node", generate_node_fn)

# Edge 연결: output_key, input_key 생략 시 StateGraph가 dict 키를 그대로 전달
graph_builder.add_edge(START, "generate_node")
graph_builder.add_edge("generate_node", "search_node")
graph_builder.add_edge("search_node", END)

# 그래프 컴파일
graph = graph_builder.compile()


# --- RAG 실행 함수 ---
def run_rag_graph(question: str) -> str:
    result = graph.invoke({"question": question, "documents": [], "answer": ""})
    return result["answer"]
