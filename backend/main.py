# FastAPI 메인 서버
# 콕스웨이브 과제 전형 - FAQ 챗봇 API

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from dotenv import load_dotenv
import os

from chatbot_service import ChatbotService

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="스마트스토어 FAQ 챗봇 API",
    description="네이버 스마트스토어 FAQ 챗봇 (OpenAI + Solar Mini 하이브리드)",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실전에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 챗봇 서비스 (전역)
chatbot_service = None


# ==================== Pydantic 모델 ====================

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    query: str = Field(..., description="사용자 질문", min_length=1, max_length=500)
    session_id: Optional[str] = Field(default="default", description="세션 ID")
    use_hybrid: Optional[bool] = Field(default=True, description="Hybrid RAG 사용 여부")
    top_k: Optional[int] = Field(default=5, description="검색할 문서 개수", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "스마트스토어 가입은 어떻게 하나요?",
                "session_id": "user_123",
                "use_hybrid": True,
                "top_k": 5
            }
        }


class ContextualQuestionRequest(BaseModel):
    """역질문 답변 요청 모델"""
    contextual_question: str = Field(..., description="역질문", min_length=1, max_length=200)
    original_query: str = Field(..., description="원래 사용자 질문", min_length=1)
    original_answer: str = Field(..., description="원래 답변", min_length=1)
    session_id: Optional[str] = Field(default="default", description="세션 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "contextual_question": "법정대리인 동의서 양식 필요하신가요?",
                "original_query": "미성년자도 등록이 가능함?",
                "original_answer": "네, 미성년자도 스마트스토어에 등록할 수 있습니다...",
                "session_id": "user_123"
            }
        }


class SourceDocument(BaseModel):
    """참고 문서 모델"""
    category: str = Field(..., description="카테고리")
    question: str = Field(..., description="질문")
    similarity: float = Field(..., description="유사도 점수")


class ContextualQuestion(BaseModel):
    """맥락 기반 역질문 모델"""
    question: str = Field(..., description="역질문")
    answer: str = Field(..., description="역질문에 대한 답변")


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    answer: str = Field(..., description="챗봇 답변")
    follow_up_questions: List[str] = Field(..., description="후속 질문 목록 (추천)")
    contextual_questions: List[ContextualQuestion] = Field(default=[], description="맥락 기반 역질문 목록")
    sources: List[SourceDocument] = Field(..., description="참고 문서 목록")
    is_smartstore_related: bool = Field(..., description="스마트스토어 관련 질문 여부")
    session_id: str = Field(..., description="세션 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "스마트스토어 가입은 네이버 커머스 ID로 간편하게 가입할 수 있습니다...",
                "follow_up_questions": [
                    "가입에 필요한 서류는 무엇인가요?",
                    "사업자등록증이 없어도 가입할 수 있나요?",
                    "가입 후 첫 상품 등록은 어떻게 하나요?"
                ],
                "contextual_questions": [
                    "등록에 필요한 서류 안내해드릴까요?",
                    "등록 절차는 얼마나 오래 걸리는지 안내가 필요하신가요?"
                ],
                "sources": [
                    {
                        "category": "가입절차",
                        "question": "스마트스토어센터 회원가입은 어떻게 하나요?",
                        "similarity": 0.95
                    }
                ],
                "is_smartstore_related": True,
                "session_id": "user_123"
            }
        }


class ConversationHistoryResponse(BaseModel):
    """대화 기록 응답 모델"""
    session_id: str = Field(..., description="세션 ID")
    history: List[Dict[str, str]] = Field(..., description="대화 기록")
    message_count: int = Field(..., description="메시지 개수")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="서버 상태")
    chat_provider: str = Field(..., description="채팅 제공자")
    total_documents: int = Field(..., description="총 문서 수")


# ==================== 이벤트 핸들러 ====================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global chatbot_service

    logger.info("=== 서버 시작 ===")
    logger.info("챗봇 서비스 초기화 중...")

    try:
        chatbot_service = ChatbotService()
        stats = chatbot_service.rag_service.get_stats()

        logger.info(f"✅ 초기화 완료")
        logger.info(f"  - 총 문서: {stats['total_documents']}개")
        logger.info(f"  - 임베딩 모델: {stats['embedding_model']}")
        logger.info(f"  - 채팅 모델: {stats['chat_model']}")
        logger.info(f"  - 채팅 제공자: {chatbot_service.chat_provider}")

    except Exception as e:
        logger.error(f"❌ 초기화 실패: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("=== 서버 종료 ===")


# ==================== API 엔드포인트 ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "스마트스토어 FAQ 챗봇 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")

    stats = chatbot_service.rag_service.get_stats()

    return HealthResponse(
        status="healthy",
        chat_provider=chatbot_service.chat_provider,
        total_documents=stats["total_documents"]
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    FAQ 챗봇 대화 (비스트리밍)

    - **query**: 사용자 질문
    - **session_id**: 세션 ID (대화 기록 관리용)
    - **use_hybrid**: Hybrid RAG 사용 여부 (기본: True)
    - **top_k**: 검색할 문서 개수 (기본: 5)
    """
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")

    try:
        result = chatbot_service.chat(
            query=request.query,
            session_id=request.session_id,
            use_hybrid=request.use_hybrid,
            top_k=request.top_k
        )

        return ChatResponse(
            answer=result["answer"],
            follow_up_questions=result["follow_up_questions"],
            contextual_questions=[
                ContextualQuestion(**cq) if isinstance(cq, dict) else ContextualQuestion(question=cq, answer="")
                for cq in result.get("contextual_questions", [])
            ],
            sources=[
                SourceDocument(**source) for source in result["sources"]
            ],
            is_smartstore_related=result["is_smartstore_related"],
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"채팅 오류: {e}")
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류 발생: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    FAQ 챗봇 대화 (스트리밍)

    - **query**: 사용자 질문
    - **session_id**: 세션 ID
    - **use_hybrid**: Hybrid RAG 사용 여부
    - **top_k**: 검색할 문서 개수

    Returns: Server-Sent Events (SSE) 스트림
    """
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")

    async def generate():
        try:
            async for chunk in chatbot_service.stream_chat(
                query=request.query,
                session_id=request.session_id,
                use_hybrid=request.use_hybrid,
                top_k=request.top_k
            ):
                # SSE 형식으로 전송
                yield f"data: {chunk}\n\n"

            # 스트림 종료 신호
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"스트리밍 오류: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/chat/contextual")
async def answer_contextual_question(request: ContextualQuestionRequest):
    """
    역질문 답변 생성 (클릭 시 호출)

    - **contextual_question**: 역질문
    - **original_query**: 원래 사용자 질문
    - **original_answer**: 원래 답변
    - **session_id**: 세션 ID

    Returns: 역질문에 대한 답변
    """
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")

    try:
        answer = chatbot_service.answer_contextual_question(
            contextual_question=request.contextual_question,
            original_query=request.original_query,
            original_answer=request.original_answer,
            session_id=request.session_id
        )

        return {
            "answer": answer,
            "contextual_question": request.contextual_question
        }

    except Exception as e:
        logger.error(f"역질문 답변 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"역질문 답변 생성 실패: {str(e)}")


@app.get("/conversation/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str):
    """
    대화 기록 조회

    - **session_id**: 세션 ID
    """
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")

    try:
        history = chatbot_service.get_conversation_history(session_id)

        return ConversationHistoryResponse(
            session_id=session_id,
            history=history,
            message_count=len(history)
        )

    except Exception as e:
        logger.error(f"대화 기록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"대화 기록 조회 실패: {str(e)}")


@app.delete("/conversation/{session_id}")
async def clear_conversation_history(session_id: str):
    """
    대화 기록 삭제

    - **session_id**: 세션 ID
    """
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")

    try:
        chatbot_service.clear_conversation_history(session_id)

        return {
            "message": f"세션 {session_id}의 대화 기록이 삭제되었습니다",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"대화 기록 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"대화 기록 삭제 실패: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    서비스 통계 조회 (RAG + 캐시 통계)
    """
    if chatbot_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")

    try:
        rag_stats = chatbot_service.rag_service.get_stats()
        cache_stats = chatbot_service.query_cache.get_cache_stats()

        return {
            "rag": rag_stats,
            "query_cache": cache_stats,  # 글로벌 캐싱 통계
            "chat_provider": chatbot_service.chat_provider,
            "total_sessions": len(chatbot_service.conversation_history),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai")
        }

    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")


# ==================== 실행 ====================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    logger.info(f"서버 시작: http://{host}:{port}")
    logger.info(f"API 문서: http://{host}:{port}/docs")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
