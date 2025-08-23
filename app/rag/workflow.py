from langgraph.graph import StateGraph
from app.rag.nodes import (
    create_vectorstore,
    generate_answer,
    load_pdf,
    retrieve_context,
    split_chunks,
)
from app.rag.state import GraphState, QueryState


# PDF Workflow 
pdf_workflow = StateGraph(GraphState)
pdf_workflow.add_node("load_pdf", load_pdf)
pdf_workflow.add_node("split_chunks", split_chunks)
pdf_workflow.add_node("create_vectorstore", create_vectorstore)

pdf_workflow.set_entry_point("load_pdf")
pdf_workflow.add_edge("load_pdf", "split_chunks")
pdf_workflow.add_edge("split_chunks", "create_vectorstore")
pdf_workflow.set_finish_point("create_vectorstore")

pdf_graph = pdf_workflow.compile()


# Query Workflow 
query_workflow = StateGraph(QueryState)
query_workflow.add_node("retrieve_context", retrieve_context)
query_workflow.add_node("generate_answer", generate_answer)

query_workflow.set_entry_point("retrieve_context")
query_workflow.add_edge("retrieve_context", "generate_answer")
query_workflow.set_finish_point("generate_answer")

query_graph = query_workflow.compile()
