# langChain_Rag

이 프로젝트는 LangChain과 RAG(Retrieval-Augmented Generation) 기반의 Python 애플리케이션입니다. PDF 문서 등에서 정보를 추출하고, LLM(대형 언어 모델)과 연동하여 질의응답 및 다양한 자연어 처리 기능을 제공합니다.

## 주요 폴더 및 파일 구조

```
pyproject.toml
README.md
uv.lock
app/
  main.py
  api/
  core/
    llm_client.py
  docs/
    iphone.pdf
  rag/
    nodes.py
    state.py
    workflow.py
```

### 주요 구성 요소
- **app/main.py**: 애플리케이션 진입점
- **app/core/llm_client.py**: LLM 연동 클라이언트
- **app/rag/**: RAG 워크플로우, 상태 관리, 노드 정의
- **app/docs/**: 예시 PDF 문서

## 설치 및 실행 방법

1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   # 또는
   pip install uv
   uv pip install -r requirements.txt
   ```

2. 애플리케이션 실행
   ```bash
   python app/main.py
   ```

## 기능 예시
- PDF 등 문서에서 정보 추출
- LLM 기반 질의응답
- RAG 워크플로우 관리

## 기여 방법
1. 이슈 등록 및 포크
2. 브랜치 생성 후 작업
3. PR(Pull Request) 요청

## 라이선스
MIT License

---

자세한 사용법 및 API 문서는 추후 업데이트 예정입니다.
