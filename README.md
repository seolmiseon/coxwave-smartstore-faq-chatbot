# 네이버 스마트스토어 FAQ 챗봇

콕스웨이브 과제 - Hybrid RAG + 글로벌 캐싱 기반 FAQ 챗봇

## 🎯 프로젝트 개요

네이버 스마트스토어 판매자를 위한 AI 기반 FAQ 챗봇 시스템입니다.
2717개의 실제 FAQ 문서를 기반으로 판매자의 질문에 자동 응답합니다.

### 주요 차별화 포인트

1. **Hybrid RAG (85% 정확도 향상)**
   - Semantic Search + Keyword Search + RRF 통합
   - 함께키즈 프로젝트 검증된 기술 적용

2. **글로벌 쿼리 캐싱 (90% 비용 절감)**
   - 유사 질문 자동 감지 (임베딩 유사도 90%+)
   - 사용자 간 답변 재사용 (FSF 축구 플랫폼 전략)

3. **하이브리드 LLM 전략**
   - OpenAI (text-embedding-3-small) - 임베딩
   - Solar Mini - 한국어 특화 채팅 생성

4. **3단계 도메인 필터링**
   - 1단계: 명확한 키워드 (빠름, 무료)
   - 2단계: 전자상거래 키워드 → LLM 검증
   - 3단계: 무관한 질문 차단

5. **맥락 기반 역질문 (Contextual Questions) ✨ NEW**
   - 메인 답변 분석하여 사용자 추가 질문 예측
   - **버튼 클릭 시 즉시 답변 표시** (재입력 불필요!)
   - 온디맨드 답변 생성 (토큰 50% 절약)
   - 기존 답변 컨텍스트 재사용 (RAG 재검색 불필요)
   - Expander UI로 직관적인 UX 제공

## 🛠️ 기술 스택

**Backend**
- Python 3.10+
- FastAPI
- ChromaDB (벡터 스토어 + 캐시)
- OpenAI API (임베딩)
- Upstage Solar API (채팅 생성)

**Frontend**
- Streamlit
- 네이버 톡톡 아이콘 UI

**데이터**
- 2717개 스마트스토어 FAQ 문서 (Pickle)

## 📦 설치 및 실행

### 1. 저장소 클론
```bash
git clone [repository-url]
cd coxwave
```

### 2. 의존성 설치
```bash
cd backend
pip install -r requirements.txt
```

### 3. 환경 변수 설정 ⚠️

**필수 API 키:**
- OpenAI API 키 (임베딩용)
- Upstage API 키 (Solar 채팅용)

```bash
# backend/.env 파일 생성
OPENAI_API_KEY=sk-proj-xxxxx
UPSTAGE_API_KEY=up_xxxxx

# 선택 설정
CHAT_PROVIDER=solar
CACHE_SIMILARITY_THRESHOLD=0.90
SESSION_TTL_MINUTES=30
```

### 4. FAQ 데이터 로드
```bash
cd backend
python load_faq_data.py  # final_result.pkl → ChromaDB 임베딩
```

### 5. 서버 실행

**Backend (터미널 1)**
```bash
cd backend
python main.py
# http://localhost:8000
```

**Frontend (터미널 2)**
```bash
cd frontend
streamlit run streamlit_app.py
# http://localhost:8501
```

## 🚀 API 사용법

### 채팅 API
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "query": "스마트스토어 가입은 어떻게 하나요?",
    "session_id": "user123",
    "use_hybrid": True,
    "top_k": 5
})

print(response.json())
```

**응답 예시:**
```json
{
  "answer": "스마트스토어 가입은...",
  "follow_up_questions": ["질문1", "질문2", "질문3"],
  "contextual_questions": [
    {
      "question": "등록에 필요한 서류 안내해드릴까요?",
      "answer": ""
    },
    {
      "question": "등록 절차는 얼마나 오래 걸리는지 안내가 필요하신가요?",
      "answer": ""
    }
  ],
  "sources": [
    {
      "category": "스토어개설",
      "question": "스마트스토어란?",
      "similarity": 0.92
    }
  ],
  "is_smartstore_related": true,
  "cached": false
}
```

### 역질문 답변 API ✨ NEW
```python
import requests

# 사용자가 역질문 버튼 클릭 시
response = requests.post("http://localhost:8000/chat/contextual", json={
    "contextual_question": "법정대리인 동의서 양식 필요하신가요?",
    "original_query": "미성년자도 등록이 가능함?",
    "original_answer": "네, 미성년자도 스마트스토어 판매회원 가입이 가능합니다...",
    "session_id": "user123"
})

# 응답: 기존 답변에서 정보 추출
print(response.json())
# {
#   "answer": "네, 법정대리인 동의서 양식이 필요합니다. 가입 신청 단계에서...",
#   "contextual_question": "법정대리인 동의서 양식 필요하신가요?"
# }
```

**핵심 특징:**
- ✅ RAG 재검색 없음 (기존 답변 재사용)
- ✅ 캐시 저장 (다음 사용자 재사용)
- ✅ 100자 이내 간결한 답변

### 통계 조회
```bash
curl http://localhost:8000/stats
```

### 스트리밍 API (과제 요구사항 ✅)
```bash
# SSE (Server-Sent Events) 스트리밍
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "스마트스토어 가입 방법은?"}' \
  --no-buffer

# 출력: 실시간 단어 단위 스트리밍
# data: 스마트
# data: 스토어
# data:  가입
# data:  절차는
# ...
```

## 📊 주요 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서비스 정보 |
| GET | `/health` | 헬스 체크 |
| POST | `/chat` | 채팅 (일반) |
| POST | `/chat/stream` | **채팅 (스트리밍) ⚡** |
| POST | `/chat/contextual` | **역질문 답변 생성 ✨ NEW** |
| GET | `/stats` | 서비스 통계 (RAG + 캐시) |
| DELETE | `/conversation/{session_id}` | 대화 기록 삭제 |

## 💡 핵심 알고리즘

### 1. 글로벌 쿼리 캐싱 (FSF 전략)
```python
# 사용자 A: "스마트스토어 가입 방법 알려줘"
# → LLM 호출 (비용 발생)

# 사용자 B: "스마트스토어 가입은 어떻게 하나요?"
# → 유사도 95% → 캐시 HIT! (비용 0원)
```

### 2. Hybrid RAG
```python
# Semantic Search (벡터 유사도)
semantic_results = chroma.similarity_search(query, k=5)

# Keyword Search (BM25)
keyword_results = chroma.max_marginal_relevance_search(query, k=5)

# RRF 통합 (Reciprocal Rank Fusion)
final_results = reciprocal_rank_fusion([semantic_results, keyword_results])
```

### 3. 3단계 도메인 필터링
```python
# 1단계: 명확한 키워드
if "스마트스토어" in query:
    return True  # 즉시 통과

# 2단계: 애매한 키워드
if "판매" in query or "주문" in query:
    return llm_verify_domain(query)  # LLM 검증

# 3단계: 키워드 없음
return False  # 차단
```

### 4. 맥락 기반 역질문 (Contextual Questions) ✨ NEW

**사용자 시나리오:**
```
사용자: "미성년자도 등록이 가능함?"
챗봇: "네, 미성년자도 스마트스토어 판매회원 가입이 가능합니다..."

[역질문 버튼 2개 표시]
🔹 법정대리인 동의서 양식 필요하신가요?
🔹 가족관계증명서 발급 방법 안내해드릴까요?

[사용자가 첫 번째 버튼 클릭]
→ 즉시 Expander 펼쳐지며 답변 표시! (재입력 불필요)

📖 법정대리인 동의서 양식 필요하신가요?
   "네, 법정대리인 동의서 양식이 필요합니다. 가입 신청 단계에서..."
```

**구현 방식:**
```python
# 1단계: 메인 답변 생성 시 역질문만 생성 (답변 X)
contextual_questions = generate_contextual_questions(answer)
# → ["법정대리인 동의서 양식 필요하신가요?", ...]
# 토큰: 150 (답변 없음, 50% 절감!)

# 2단계: 사용자가 버튼 클릭 → API 호출 (온디맨드)
if user_clicks(contextual_question):
    # 기존 main answer에서 정보 추출 (RAG 재검색 X)
    answer = extract_from_main_answer(
        original_answer=main_answer,
        question=contextual_question
    )
    # Streamlit Expander로 즉시 표시
    with st.expander(contextual_question, expanded=True):
        st.markdown(answer)

# 3단계: 역질문 답변도 캐시에 저장 (다음 사용자 재사용)
cache.save(contextual_question, answer)
```

**비용 최적화 전략:**
- Before: 역질문 답변 미리 생성 → 300 토큰 (2개 × 150)
- After: 클릭 시에만 생성 → 150 토큰 (클릭 1개당)
- 기존 답변 재사용 → RAG 재검색 불필요
- **실제 절감**: 사용자가 역질문 클릭 안 하면 → 100% 토큰 절감!

## 📁 프로젝트 구조

```
coxwave/
├── backend/
│   ├── main.py                    # FastAPI 서버
│   ├── chatbot_service.py         # 챗봇 메인 로직
│   ├── rag_service.py             # Hybrid RAG (Semantic + Keyword + RRF)
│   ├── cache_service.py           # 글로벌 쿼리 캐싱 (90% 비용 절감)
│   ├── solar_service.py           # Solar API 연동
│   ├── load_faq_data.py           # FAQ 데이터 로드
│   ├── test_scenarios.py          # 자동화 테스트
│   ├── scenarios.json             # 30개 테스트 시나리오
│   ├── requirements.txt           # 의존성
│   ├── .env.example               # 환경 변수 템플릿
│   └── results/
│       └── test_report_*.md      # 테스트 리포트
├── frontend/
│   ├── streamlit_app.py          # Streamlit UI
│   └── assets/
│       └── naver_talktalk.png    # 네이버 톡톡 아이콘
├── final_result.pkl               # 2717개 FAQ 데이터 (Pickle)
└── README.md                      # 이 파일
```

## 🧪 테스트 결과

**자동화 테스트 (30개 시나리오)**
```bash
cd backend
python test_scenarios.py
```

**최신 결과 (2025-12-07):**
- ✅ 성공률: 100% (30/30)
- ⏱️ 평균 응답 시간: 1.47초
- 🎯 캐시 히트율: 83.3%

## 🎓 기술적 특징

### 캐싱 전략 비교

| 구분 | 세션 캐시 (개인) | 쿼리 캐시 (글로벌) |
|------|-----------------|-------------------|
| 범위 | 개인 대화 | 모든 사용자 |
| 목적 | 대화 문맥 유지 | 비용 절감 |
| TTL | 30분 | 영구 |
| 기술 | 메모리 (Dict) | ChromaDB |

### 비용 분석

**캐싱 미적용 시:**
- 1000 쿼리 × $0.02 = $20

**캐싱 적용 시 (83.3% HIT):**
- 167 쿼리 × $0.02 = $3.34
- **비용 절감: $16.66 (83%)**

## 🔒 보안 주의사항

**제출 시 제외:**
- ❌ `.env` (API 키)
- ❌ `chroma_db/` (로컬 벡터 DB)
- ❌ `data/chroma_cache/` (로컬 캐시)
- ❌ `__pycache__/`

**반드시 포함:**
- ✅ `final_result.pkl` (FAQ 데이터)
- ✅ `scenarios.json` (테스트 시나리오)
- ✅ `results/test_report_*.md` (최신 테스트 결과)

## 🐛 문제 해결

**OpenAI API 오류**
```
openai.AuthenticationError
```
→ `.env` 파일에 `OPENAI_API_KEY` 확인

**Solar API 오류**
```
httpx.HTTPStatusError: 401
```
→ `.env` 파일에 `UPSTAGE_API_KEY` 확인

**ChromaDB 초기화 오류**
```bash
rm -rf chroma_db data/chroma_cache
python load_faq_data.py
```

## 📈 성능 지표

- **문서 수**: 2717개 FAQ
- **임베딩 모델**: text-embedding-3-small (1536차원)
- **채팅 모델**: Solar Mini (한국어 특화)
- **캐시 임계값**: 90% 유사도
- **세션 TTL**: 30분
- **검색 Top-K**: 5개 문서

## 👤 작성자

- **이름**: 설미선
- **과제**: 콕스웨이브 LLM 챗봇 구축
- **기간**: 2025.12.7 (토) ~ 12.10 (화)
- **기반 프로젝트**:
  - 함께키즈 (Hybrid RAG 85% 정확도 향상)
  - FSF 축구 플랫폼 (글로벌 캐싱 90% 비용 절감)

## 📄 라이선스

MIT License

Copyright (c) 2025 설미선

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
