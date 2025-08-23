from app.core import llm_client
from app.rag.state import GraphState, QueryState

# 랭체인
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.rag.state import GraphState, QueryState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# PDF 가져오기
def load_pdf(state: GraphState) -> GraphState:
    loader = PyPDFLoader(state["file_path"])
    state["documents"] = loader.load()
    return state


# 문서 청크
def split_chunks(state: GraphState) -> GraphState:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    state["chunks"] = splitter.split_documents(state.get("documents", []))
    return state


# 임베딩, 벡터스토어 생성(FAISS)
def create_vectorstore(state: GraphState) -> GraphState:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    state["vectorstore"] = FAISS.from_documents(state.get("documents", []), embeddings)
    return state


# 질문에 대한 context 검색
def retrieve_context(state: QueryState) -> QueryState:
    vectorstore = state["vectorstore"]  # state 안에서 꺼냄
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(state["question"])
    state["context"] = "\n\n".join([doc.page_content for doc in docs])
    return state


# context 기반 답변 생성
def generate_answer(state: QueryState) -> QueryState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "Answer strictly based on the provided context. "
                "If not found, say '관련 정보를 찾을 수 없습니다.'",
            ),
            ("user", "Question: {question}\nContext:\n{context}"),
        ]
    )

    chain = prompt | llm_client.llm | StrOutputParser()
    answer = chain.invoke(
        {"question": state["question"], "context": state.get("context", [])}
    )
    state["answer"] = answer
    return state
