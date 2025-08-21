import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from app.core import llm_client
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import StrOutputParser
from app.rag.state import GraphState
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

load_dotenv()


def search_node_fn(state: GraphState) -> GraphState:
    question = state["question"]

    # PDF 경로 가져오기
    PDF_PATH = os.path.abspath("app/docs/iphone.pdf")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # --- 문서 청크 ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # --- 벡터 DB 생성 (FAISS) ---
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # --- 질문과 유사한 문서 검색 ---
    retrived_docs = vectorstore.similarity_search(question, k=3)

    state["documents"] = retrived_docs
    return state


def generate_node_fn(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]

    # 문서 내용 합치기
    context = "\n".join([doc.page_content for doc in documents])

    # Prompt 정의
    prompt = ChatPromptTemplate.from_template(
        "다음 문서를 참고하여 질문에 답하세요:\n{context}\n질문: {question}\n답변:"
    )

    # LLMChain 생성
    llm_chain = prompt | llm_client.llm | StrOutputParser()

    # 최신 방식: predict 사용
    answer = llm_chain.invoke({"context": context, "question": question})
    state["answer"] = answer
    return state
