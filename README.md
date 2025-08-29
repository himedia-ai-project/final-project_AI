# langChain_Rag

이 프로젝트는 LangChain과 RAG(Retrieval-Augmented Generation) 기반의 Python 애플리케이션입니다. PDF 문서 등에서 정보를 추출하고, LLM(대형 언어 모델)과 연동하여 질의응답 및 다양한 자연어 처리 기능을 제공합니다.

## 주요 폴더 및 파일 구조

```

├── pyproject.toml      # 프로젝트 의존성 및 패키지 설정
├── README.md           # 프로젝트 문서
├── uv.lock             # uv 패키지 관리 Lock 파일
├── app/
│   ├── main.py         # FastAPI 진입점 (서버 실행)
│   ├── api/            # API 라우터 및 엔드포인트
│   ├── core/
│   │   └── llm\_client.py  # LLM 클라이언트 설정 (OpenAI 등)
│   ├── docs/
│   │   └── iphone.pdf  # 예시 문서 (RAG 학습 데이터)
│   └── rag/
│       ├── nodes.py    # LangGraph 노드 정의
│       ├── state.py    # 상태 관리 (RAG 상태 저장)
│       └── workflow\.py # 전체 워크플로우 정의

```

### 주요 구성 요소
- **app/main.py**: 애플리케이션 진입점
- **app/core/llm_client.py**: LLM 연동 클라이언트
- **app/rag/**: RAG 워크플로우, 상태 관리, 노드 정의
- **app/docs/**: 예시 PDF 문서

## 설치 및 실행 방법

1. 의존성 설치
   ```bash
  
   pip install uv
   uv pip install -r requirements.txt
   ```

2. 애플리케이션 실행
   ```bash
   uv run uvicorn app.main:app --reload
   ```

## 기능 예시
- PDF 등 문서에서 정보 추출
- LLM 기반 질의응답
- RAG 워크플로우 관리

---

