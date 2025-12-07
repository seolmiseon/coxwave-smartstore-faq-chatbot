# Streamlit 프론트엔드 (콕스웨이브 과제용)
# 20분 만에 완성되는 채팅 UI

import streamlit as st
import requests
import sys
import os
import base64
import time
from pathlib import Path
from PIL import Image
from io import BytesIO

# Backend 경로 추가 (직접 임포트 가능)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# 이미지를 리사이즈하고 base64로 인코딩하는 함수
def get_resized_image_base64(image_path, size=(80, 80)):
    """이미지를 리사이즈하고 base64 문자열로 변환"""
    img = Image.open(image_path)
    img = img.resize(size, Image.Resampling.LANCZOS)

    # PNG로 저장 (투명 배경 유지)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

# 네이버 톡톡 아이콘 경로
NAVER_ICON_PATH = Path(__file__).parent / "assets" / "naver_talktalk.png"

# 페이지 설정
st.set_page_config(
    page_title="네이버 스마트스토어 FAQ 챗봇",
    page_icon="🛒",  # 스마트스토어 = 쇼핑
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")

    # API 엔드포인트 설정
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="FastAPI 백엔드 주소"
    )

    # RAG 모드 선택
    use_hybrid = st.checkbox("🔀 Hybrid RAG 사용", value=True, help="Semantic + Keyword 검색 통합")
    top_k = st.slider("📚 검색 문서 수", min_value=1, max_value=10, value=5)

    # 스트리밍 모드 선택
    use_streaming = st.checkbox("⚡ 스트리밍 모드 (실험적)", value=False, help="API는 지원하나 UI 한계로 OFF 권장")

    # 세션 ID
    session_id = st.text_input("👤 세션 ID", value="default", help="대화 기록 관리용")

    st.divider()

    # 통계 정보
    st.subheader("📊 서비스 통계")
    if st.button("통계 조회"):
        try:
            response = requests.get(f"{api_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                st.json(stats)
            else:
                st.error("통계 조회 실패")
        except:
            st.error("API 연결 실패")

# 메인 화면
st.title("🛒 네이버 스마트스토어 FAQ 챗봇")
st.caption("💚 네이버 스마트스토어 판매자를 위한 AI FAQ 도우미 | OpenAI + Solar Mini 하이브리드")

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# 채팅 히스토리 표시
for message in st.session_state.messages:
    # 아바타 설정: 사용자는 쇼핑카트, 봇은 네이버 톡톡 로고
    if message["role"] == "user":
        avatar = "🛒"
    else:
        # 네이버 톡톡 아이콘 사용
        if NAVER_ICON_PATH.exists():
            avatar = f"data:image/png;base64,{get_resized_image_base64(NAVER_ICON_PATH, size=(64, 64))}"
        else:
            avatar = "💚"  # 폴백

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # 후속 질문 표시
        if message.get("follow_up_questions"):
            st.caption("💡 추천 질문:")
            for fq in message["follow_up_questions"]:
                st.caption(f"  • {fq}")

# 채팅 입력
# pending_question이 있으면 먼저 처리
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None  # 초기화
elif prompt := st.chat_input("질문을 입력하세요"):
    pass  # prompt는 이미 설정됨
else:
    prompt = None

if prompt:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # API 호출
    # 네이버 톡톡 아이콘으로 assistant 메시지 표시
    naver_avatar = f"data:image/png;base64,{get_resized_image_base64(NAVER_ICON_PATH, size=(64, 64))}" if NAVER_ICON_PATH.exists() else "💚"
    with st.chat_message("assistant", avatar=naver_avatar):
        message_placeholder = st.empty()

        try:
            # 요청 데이터 구성
            payload = {
                "query": prompt,
                "session_id": session_id,
                "use_hybrid": use_hybrid,
                "top_k": top_k
            }

            # 스트리밍 모드
            if use_streaming:
                # SSE 스트리밍
                response = requests.post(
                    f"{api_url}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=30
                )

                if response.status_code == 200:
                    full_answer = ""
                    # decode_unicode=True로 실시간 디코딩
                    for line in response.iter_lines(decode_unicode=True, delimiter='\n'):
                        if line and line.startswith("data: "):
                            chunk = line[6:]  # "data: " 제거

                            if chunk == "[DONE]":
                                break
                            if chunk.startswith("[ERROR]"):
                                st.error(chunk)
                                break

                            full_answer += chunk
                            # 실시간 업데이트
                            message_placeholder.markdown(full_answer + " ▌")

                    # 최종 답변 (커서 제거)
                    message_placeholder.markdown(full_answer)

                    # 일반 응답으로 후속 질문/역질문/참고 문서 가져오기
                    detail_response = requests.post(
                        f"{api_url}/chat",
                        json=payload,
                        timeout=10
                    )
                    if detail_response.status_code == 200:
                        data = detail_response.json()
                        follow_ups = data.get("follow_up_questions", [])
                        contextual_questions = data.get("contextual_questions", [])  # 역질문
                        sources = data.get("sources", [])

                        # 역질문 표시 (답변 바로 뒤 - 클릭하면 답변 펼치기)
                        if contextual_questions:
                            st.markdown("---")
                            for idx, cq_data in enumerate(contextual_questions):
                                cq_question = cq_data.get("question", cq_data) if isinstance(cq_data, dict) else cq_data
                                cq_answer = cq_data.get("answer", "") if isinstance(cq_data, dict) else ""

                                if cq_answer:
                                    # 답변이 있으면 expander로 표시
                                    with st.expander(f"💬 {cq_question}"):
                                        st.markdown(cq_answer)
                                else:
                                    # 답변이 없으면 기존 방식 (버튼)
                                    if st.button(cq_question, key=f"stream_contextual_{idx}_{len(st.session_state.messages)}"):
                                        st.session_state.pending_question = cq_question
                                        st.rerun()
                    else:
                        follow_ups = []
                        contextual_questions = []
                        sources = []

                    answer = full_answer

                else:
                    error_msg = f"API 오류: {response.status_code}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.stop()

            # 일반 모드 (기존)
            else:
                response = requests.post(
                    f"{api_url}/chat",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "응답을 받지 못했습니다.")
                    follow_ups = data.get("follow_up_questions", [])
                    contextual_questions = data.get("contextual_questions", [])  # 역질문 추가
                    sources = data.get("sources", [])
                    is_related = data.get("is_smartstore_related", True)

                    # 응답 표시
                    message_placeholder.markdown(answer)

                    # 역질문 표시 (클릭 시 답변 생성)
                    if contextual_questions:
                        st.markdown("---")
                        st.markdown("**💬 추가로 궁금하신 내용**")

                        # 세션에 역질문 답변 캐시 저장
                        if "contextual_answers" not in st.session_state:
                            st.session_state.contextual_answers = {}

                        for idx, cq in enumerate(contextual_questions):
                            cq_question = cq if isinstance(cq, str) else cq.get("question", "")
                            button_key = f"contextual_{idx}_{len(st.session_state.messages)}"

                            # 버튼 클릭 시 답변 생성
                            if st.button(f"🔹 {cq_question}", key=button_key):
                                # API 호출하여 답변 생성
                                try:
                                    cq_response = requests.post(
                                        f"{api_url}/chat/contextual",
                                        json={
                                            "contextual_question": cq_question,
                                            "original_query": prompt,
                                            "original_answer": answer,
                                            "session_id": session_id
                                        },
                                        timeout=15
                                    )

                                    if cq_response.status_code == 200:
                                        cq_answer = cq_response.json().get("answer", "")
                                        # 세션에 저장
                                        st.session_state.contextual_answers[button_key] = cq_answer
                                        st.rerun()
                                    else:
                                        st.error(f"역질문 답변 생성 실패: {cq_response.status_code}")

                                except Exception as e:
                                    st.error(f"역질문 처리 오류: {str(e)}")

                            # 이미 답변이 있으면 expander로 표시
                            if button_key in st.session_state.contextual_answers:
                                with st.expander(f"📖 {cq_question}", expanded=True):
                                    st.markdown(st.session_state.contextual_answers[button_key])

                else:
                    error_msg = f"API 오류: {response.status_code}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.stop()

            # 참고 문서 표시 (스트리밍/일반 공통)
            if sources:
                with st.expander("📚 참고 문서"):
                    for i, src in enumerate(sources, 1):
                        st.caption(f"{i}. [{src['category']}] {src['question']}")

            # 후속 질문 표시 (스트리밍/일반 공통)
            if follow_ups:
                st.caption("💡 추천 질문:")
                for fq in follow_ups:
                    st.caption(f"  • {fq}")

            # 통계 업데이트
            st.session_state.total_queries += 1

            # 메시지 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "follow_up_questions": follow_ups,
                "sources": sources
            })

        except requests.exceptions.ConnectionError:
            error_msg = "❌ API 서버에 연결할 수 없습니다. Backend가 실행 중인지 확인하세요."
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

        except Exception as e:
            error_msg = f"오류 발생: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# 하단 통계 표시
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.metric("💬 전체 질문 수", st.session_state.total_queries)

with col2:
    st.metric("🤖 챗봇 모델", "Solar Mini (한국어 특화)")

# 채팅 초기화 버튼
if st.button("🔄 대화 초기화"):
    st.session_state.messages = []
    st.session_state.total_queries = 0
    # API에도 대화 기록 삭제 요청
    try:
        requests.delete(f"{api_url}/conversation/{session_id}")
    except:
        pass
    st.rerun()
