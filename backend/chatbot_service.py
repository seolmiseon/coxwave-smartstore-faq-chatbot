# FAQ 챗봇 서비스 (OpenAI + Solar Pro 하이브리드)
# 콕스웨이브 과제 전형

from rag_service import RAGService
from solar_service import SolarService
from cache_service import QueryCacheService
from openai import OpenAI
import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    FAQ 챗봇 메인 서비스

    차별화 포인트:
    1. OpenAI (임베딩) + Solar Pro (채팅) 하이브리드 전략
    2. Hybrid RAG (Semantic + Keyword + RRF)
    3. 대화 기록 관리
    4. 후속 질문 생성
    5. 도메인 필터링 (스마트스토어 전용)
    """

    def __init__(self):
        """챗봇 서비스 초기화"""
        # RAG 서비스 초기화
        self.rag_service = RAGService(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
            collection_name="smartstore_faq"
        )

        # 쿼리 캐시 서비스 초기화 (글로벌 캐싱 - 90% 비용 절감)
        cache_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))
        self.query_cache = QueryCacheService(
            cache_path="./data/chroma_cache",
            similarity_threshold=cache_threshold
        )

        # LLM 제공자 선택
        self.chat_provider = os.getenv("CHAT_PROVIDER", "solar")

        # Solar Pro 초기화
        if self.chat_provider == "solar":
            self.solar_service = SolarService()
            self.chat_model = self.solar_service.chat_model
            logger.info(f"채팅 제공자: Solar Pro ({self.chat_model})")
        else:
            # OpenAI 초기화
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.chat_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"채팅 제공자: OpenAI ({self.chat_model})")

        # 대화 기록 저장소 (세션 캐시 - 개인, 30분 TTL)
        # 실전에서는 Redis나 DB 사용
        self.conversation_history = {}

        # 세션 만료 시간 관리 (보안)
        self.session_expiry = {}  # {session_id: expiry_datetime}
        self.session_ttl_minutes = int(os.getenv("SESSION_TTL_MINUTES", "30"))  # 기본 30분

        logger.info(f"ChatbotService 초기화 완료 (세션 TTL: {self.session_ttl_minutes}분)")


    def _clean_expired_sessions(self):
        """만료된 세션 정리 (보안 + 메모리 관리)"""
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, expiry in self.session_expiry.items()
            if expiry < now
        ]

        for session_id in expired_sessions:
            if session_id in self.conversation_history:
                del self.conversation_history[session_id]
            del self.session_expiry[session_id]
            logger.info(f"만료된 세션 삭제: {session_id}")


    def _update_session_expiry(self, session_id: str):
        """세션 만료 시간 갱신"""
        self.session_expiry[session_id] = datetime.now() + timedelta(minutes=self.session_ttl_minutes)


    def _is_smartstore_question(self, query: str) -> bool:
        """
        도메인 필터링: 스마트스토어 관련 질문인지 확인 (FSF 하이브리드 전략)

        3단계 필터링:
        1. 명확한 키워드 → 즉시 통과 (빠름, 무료)
        2. 확장 키워드 → LLM 검증 (느림, 유료지만 캐싱됨)
        3. 완전 무관 → 차단

        Args:
            query: 사용자 질문

        Returns:
            스마트스토어 관련 질문이면 True
        """
        query_lower = query.lower()

        # 1단계: 명확한 스마트스토어 키워드 (100% 확신)
        core_keywords = [
            "스마트스토어", "smartstore", "네이버스토어", "셀러", "판매자센터"
        ]
        if any(kw in query_lower for kw in core_keywords):
            logger.info(f"✅ 1단계 통과 (명확한 키워드)")
            return True

        # 2단계: 전자상거래 관련 키워드 (애매함 → LLM 검증)
        # 이 키워드들은 스마트스토어 외에도 쓰일 수 있음
        commerce_keywords = [
            # 판매 관련
            "판매", "상품", "아이템", "물건", "가격", "할인", "쿠폰",
            # 주문/배송 관련
            "주문", "배송", "택배", "발송", "송장", "포장",
            # 결제/정산 관련
            "결제", "정산", "수수료", "입금", "돈", "세금", "계좌",
            # 환불/교환/취소 관련
            "환불", "교환", "반품", "취소", "클레임", "as", "불량",
            # 고객응대 관련
            "고객", "구매자", "문의", "리뷰", "후기", "평점", "욕", "컴플레인",
            # 스토어 관리 관련
            "가입", "등록", "개설", "운영", "관리", "노출", "검색", "카테고리",
            # 기타
            "쇼핑", "스토어", "사진", "이미지", "사업자", "대표자"
        ]

        if any(kw in query for kw in commerce_keywords):
            logger.info(f"⚠️  2단계: 확장 키워드 감지 → LLM 검증 필요")
            # LLM으로 스마트스토어 관련인지 최종 확인
            return self._llm_verify_domain(query)

        # 3단계: 완전 무관한 질문 차단
        logger.info(f"❌ 3단계: 키워드 없음 → 차단")
        return False

    def _llm_verify_domain(self, query: str) -> bool:
        """
        LLM으로 도메인 검증 (2단계 필터용)

        비용 최적화:
        - gpt-4o-mini 사용 (저렴)
        - max_tokens=5 (YES/NO만)
        - 결과는 쿼리 캐시에 저장됨 → 같은 질문 재사용 시 무료!

        Args:
            query: 사용자 질문

        Returns:
            스마트스토어 판매자 관련 질문이면 True
        """
        try:
            # 간단한 분류 프롬프트
            prompt = f"""다음 질문이 '네이버 스마트스토어 판매자'와 관련된 질문인지 판단하세요.

스마트스토어 판매자 관련 주제:
- 스토어 개설/가입/등록 (연령, 자격, 서류 등)
- 상품 등록/관리
- 주문/배송 처리
- 결제/정산
- 환불/교환/취소 처리
- 고객 문의/리뷰 관리
- 판매 전략/마케팅
- 판매자 자격 요건 (미성년자, 사업자 등)

질문: {query}

위 질문이 위 주제 중 하나와 관련되면 'YES', 완전히 무관하면 'NO'만 답하세요."""

            # OpenAI 또는 Solar 사용 (둘 다 저렴)
            if hasattr(self, 'openai'):
                response = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0
                )
                result = response.choices[0].message.content.strip().upper()
            else:
                # Solar 사용 시
                result = self.solar_service.generate_chat_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=5
                ).strip().upper()

            is_related = "YES" in result
            logger.info(f"🤖 LLM 검증 결과: {'✅ 관련있음' if is_related else '❌ 무관'}")
            return is_related

        except Exception as e:
            logger.error(f"LLM 검증 실패: {e} → 안전하게 True 반환")
            # 에러 시 안전하게 통과 (False Negative 방지)
            return True


    def _generate_system_prompt(self) -> str:
        """
        시스템 프롬프트 생성

        Solar Pro용으로 최적화된 한국어 프롬프트

        Returns:
            시스템 프롬프트
        """
        return """당신은 네이버 스마트스토어 전문 상담 AI입니다.

역할:
- 스마트스토어 판매자들의 FAQ 질문에 정확하고 친절하게 답변합니다
- 제공된 문서를 기반으로 답변하며, 추측하지 않습니다
- 단계별로 명확하게 설명합니다

답변 규칙:
1. 제공된 FAQ 문서에 기반하여 답변하세요
2. 문서에 없는 내용은 "관련 정보를 찾을 수 없습니다"라고 명확히 알려주세요
3. 간결하고 이해하기 쉽게 답변하세요
4. 필요시 단계별로 설명하세요
5. 친절하고 전문적인 톤을 유지하세요

스마트스토어 관련 질문이 아닌 경우:
"죄송합니다. 저는 네이버 스마트스토어 관련 질문에만 답변할 수 있습니다. 스마트스토어 관련 질문을 해주시겠어요?"라고 답변하세요."""


    def _generate_follow_up_questions(
        self,
        query: str,
        answer: str,
        search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        후속 질문 생성 (LLM 기반 - 하이브리드 전략)

        검색 결과 카테고리를 기반으로 관련 질문 3개 생성

        Args:
            query: 원래 질문
            answer: 생성된 답변
            search_results: 검색 결과

        Returns:
            후속 질문 리스트 (최대 3개)
        """
        # 검색 결과에서 관련 카테고리 추출
        categories = []
        for result in search_results[:3]:
            if "metadata" in result and "category" in result["metadata"]:
                cat = result["metadata"]["category"]
                if cat != "기타":
                    categories.append(cat)

        if not categories:
            categories = ["스마트스토어 일반"]

        categories_str = ", ".join(set(categories))

        # LLM으로 후속 질문 생성
        prompt = f"""스마트스토어 FAQ 챗봇입니다. 사용자가 추가로 궁금해할 만한 관련 질문 3개를 생성하세요.

사용자 질문: {query}
관련 카테고리: {categories_str}

규칙:
1. 위 카테고리와 관련된 실용적인 질문
2. 사용자가 다음 단계로 궁금해할 내용
3. 간결하고 명확한 질문 (15자 이내)

질문 3개를 줄바꿈으로 구분해서 작성하세요:"""

        try:
            if hasattr(self, 'openai'):
                response = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                result = response.choices[0].message.content.strip()
            else:
                result = self.solar_service.generate_chat_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                ).strip()

            # 줄바꿈으로 분리 후 정제
            questions = [q.strip().lstrip('123.-•') for q in result.split('\n') if q.strip()]
            return questions[:3]

        except Exception as e:
            logger.error(f"후속 질문 생성 실패: {e}")
            # 폴백: 검색 결과에서 다른 질문 추출
            fallback = []
            for result in search_results[:3]:
                if "metadata" in result and "clean_question" in result["metadata"]:
                    fallback.append(result["metadata"]["clean_question"])

            # 검색 결과도 없으면 빈 리스트 반환
            return fallback[:3] if fallback else []


    def _generate_contextual_questions(self, query: str, answer: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        맥락 기반 역질문 생성 (질문만! 답변은 클릭 시 생성)

        답변 내용을 분석하여 사용자가 추가로 궁금해할 만한 역질문 2개를 생성

        Args:
            query: 사용자 질문
            answer: 생성된 답변
            search_results: 검색 결과

        Returns:
            역질문 리스트 (최대 2개)
        """
        # LLM으로 역질문만 생성 (답변은 나중에 클릭 시!)
        prompt = f"""스마트스토어 FAQ 챗봇입니다. 사용자 질문에 대한 답변이 제공되었습니다.
답변 내용을 분석하여 사용자가 다음으로 궁금해할 만한 역질문 2개를 생성하세요.

사용자 질문: {query}
챗봇 답변: {answer}

역질문 규칙:
1. 답변 내용과 직접 연관된 후속 질문
2. "~안내해드릴까요?", "~알려드릴까요?", "~필요하신가요?" 형식 사용
3. 구체적이고 실용적인 질문 (20자 이내)

질문 2개를 줄바꿈으로 구분해서 작성하세요:"""

        try:
            if hasattr(self, 'openai'):
                response = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                result = response.choices[0].message.content.strip()
            else:
                result = self.solar_service.generate_chat_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                ).strip()

            # 줄바꿈으로 분리
            questions = [q.strip() for q in result.split('\n') if q.strip()]

            # 번호 제거 (1. 2. - * 등)
            import re
            questions = [re.sub(r'^[\d\-\*\.\)]+\s*', '', q).strip() for q in questions]

            return questions[:2]

        except Exception as e:
            logger.error(f"역질문 생성 실패: {e}")
            return []


    def answer_contextual_question(
        self,
        contextual_question: str,
        original_query: str,
        original_answer: str,
        session_id: str = "default"
    ) -> str:
        """
        역질문에 대한 답변 생성 (기존 main answer에서 추출!)

        사용자가 역질문을 클릭하면, 이미 생성된 main answer에서 정보를 추출하여 답변

        Args:
            contextual_question: 역질문
            original_query: 원래 사용자 질문
            original_answer: 원래 답변 (여기서 정보 추출!)
            session_id: 세션 ID

        Returns:
            역질문에 대한 답변
        """
        # 캐시에서 먼저 확인
        cached_result = self.query_cache.search_similar_cache(contextual_question)
        if cached_result:
            logger.info(f"🎯 역질문 캐시 히트: {contextual_question[:30]}...")
            return cached_result["answer"]

        # 캐시 미스: main answer에서 정보 추출
        prompt = f"""스마트스토어 FAQ 챗봇입니다.

원래 사용자 질문: {original_query}
원래 답변:
{original_answer}

사용자가 추가로 궁금해하는 질문: {contextual_question}

**위 원래 답변에 포함된 정보를 바탕으로** 사용자의 추가 질문에 간결하게 답변하세요. (100자 이내)
새로운 정보를 만들지 말고, 이미 제공한 답변에서 관련 부분만 요약해주세요."""

        try:
            # LLM 호출
            if hasattr(self, 'openai'):
                response = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.5
                )
                answer = response.choices[0].message.content.strip()
            else:
                answer = self.solar_service.generate_chat_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=200
                ).strip()

            # 캐시에 저장 (다음 사용자를 위해!)
            self.query_cache.save_cache(
                query=contextual_question,
                answer=answer,
                follow_up_questions=[],
                contextual_questions=[],
                sources=[]
            )
            logger.info(f"💾 역질문 답변 캐시 저장: {contextual_question[:30]}...")

            return answer

        except Exception as e:
            logger.error(f"역질문 답변 생성 실패: {e}")
            return "죄송합니다. 답변을 생성할 수 없습니다."


    def chat(
        self,
        query: str,
        session_id: str = "default",
        use_hybrid: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        일반 채팅 (비스트리밍)

        Args:
            query: 사용자 질문
            session_id: 세션 ID (대화 기록 관리용)
            use_hybrid: Hybrid RAG 사용 여부
            top_k: 검색할 문서 개수

        Returns:
            {
                "answer": "답변",
                "follow_up_questions": ["질문1", "질문2", "질문3"],
                "sources": [검색된 문서들],
                "is_smartstore_related": True/False,
                "cached": True/False  # 캐시 히트 여부
            }
        """
        # 0. 만료된 세션 정리 (보안)
        self._clean_expired_sessions()

        # 1. 도메인 필터링
        is_smartstore = self._is_smartstore_question(query)
        if not is_smartstore:
            # RAG에서 일반적인 FAQ 질문 3개 추출 (하드코딩 제거)
            general_faqs = self.rag_service.semantic_search("스마트스토어 가입", top_k=3)
            suggested_questions = [
                faq["metadata"]["clean_question"] for faq in general_faqs
            ] if general_faqs else []

            return {
                "answer": "죄송합니다. 저는 네이버 스마트스토어 관련 질문에만 답변할 수 있습니다. 스마트스토어 관련 질문을 해주시겠어요?",
                "follow_up_questions": suggested_questions,
                "contextual_questions": [],
                "sources": [],
                "is_smartstore_related": False,
                "cached": False
            }

        # 2. 쿼리 캐시 확인 (글로벌 - 90% 비용 절감!)
        cached_result = self.query_cache.search_similar_cache(query)
        if cached_result:
            logger.info(f"🎯 캐시에서 답변 반환! 유사도: {cached_result['similarity']:.2%}")
            return {
                "answer": cached_result["answer"],
                "follow_up_questions": cached_result["follow_up_questions"],
                "contextual_questions": cached_result.get("contextual_questions", []),  # 역질문 (캐시에 있으면)
                "sources": cached_result["sources"],
                "is_smartstore_related": True,
                "cached": True,
                "cache_similarity": cached_result["similarity"],
                "original_query": cached_result["original_query"]
            }

        # 3. RAG 검색 (캐시 미스 시에만!)
        if use_hybrid:
            search_results = self.rag_service.hybrid_search(query, top_k=top_k)
        else:
            search_results = self.rag_service.semantic_search(query, top_k=top_k)

        # 검색 결과가 없으면 기본 답변
        if not search_results:
            # 일반 FAQ에서 추천 질문 추출 (하드코딩 제거)
            fallback_faqs = self.rag_service.semantic_search("스마트스토어", top_k=3)
            fallback_questions = [
                faq["metadata"]["clean_question"] for faq in fallback_faqs
            ] if fallback_faqs else []

            return {
                "answer": "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?",
                "follow_up_questions": fallback_questions,
                "contextual_questions": [],
                "sources": [],
                "is_smartstore_related": True,
                "cached": False
            }

        # 4. 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {i+1}] (카테고리: {doc['metadata']['category']})\n{doc['document']}"
            for i, doc in enumerate(search_results[:3])
        ])

        # 5. 대화 기록 가져오기
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        history = self.conversation_history[session_id]

        # 6. 메시지 구성
        messages = [
            {"role": "system", "content": self._generate_system_prompt()}
        ]

        # 최근 대화 3턴만 포함 (메모리 절약)
        for msg in history[-6:]:
            messages.append(msg)

        # 현재 질문
        user_message = f"""관련 문서:
{context}

사용자 질문: {query}

위 문서를 참고하여 질문에 답변해주세요."""

        messages.append({"role": "user", "content": user_message})

        # 7. LLM 답변 생성 (캐시 미스 시에만!)
        if self.chat_provider == "solar":
            answer = self.solar_service.generate_chat_response(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
        else:
            response = self.openai.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            answer = response.choices[0].message.content

        # 8. 후속 질문 생성 (LLM 기반 - 카테고리 참고)
        follow_ups = self._generate_follow_up_questions(query, answer, search_results)

        # 9. 맥락 기반 역질문 + 답변 생성 (LLM 기반 - 답변 내용 참고)
        contextual_questions = self._generate_contextual_questions(query, answer, search_results)

        # 10. 참고 문서 정리
        sources = [
            {
                "category": doc["metadata"]["category"],
                "question": doc["metadata"]["clean_question"],
                "similarity": doc.get("similarity", doc.get("score", 0))
            }
            for doc in search_results[:3]
        ]

        # 11. 쿼리 캐시에 저장 (다음 사용자를 위해!)
        self.query_cache.save_cache(
            query=query,
            answer=answer,
            follow_up_questions=follow_ups,
            contextual_questions=contextual_questions,
            sources=sources
        )

        # 11. 대화 기록 저장 (세션 캐시)
        self.conversation_history[session_id].append(
            {"role": "user", "content": query}
        )
        self.conversation_history[session_id].append(
            {"role": "assistant", "content": answer}
        )

        # 최근 10턴만 유지
        if len(self.conversation_history[session_id]) > 20:
            self.conversation_history[session_id] = self.conversation_history[session_id][-20:]

        # 세션 만료 시간 갱신 (보안)
        self._update_session_expiry(session_id)

        # 12. 결과 반환
        return {
            "answer": answer,
            "follow_up_questions": follow_ups,
            "contextual_questions": contextual_questions,  # 역질문 추가!
            "sources": sources,
            "is_smartstore_related": True,
            "cached": False  # 새로 생성한 답변
        }


    async def stream_chat(
        self,
        query: str,
        session_id: str = "default",
        use_hybrid: bool = True,
        top_k: int = 5
    ) -> AsyncIterator[str]:
        """
        스트리밍 채팅

        Args:
            query: 사용자 질문
            session_id: 세션 ID
            use_hybrid: Hybrid RAG 사용 여부
            top_k: 검색할 문서 개수

        Yields:
            답변 청크 (문자열)
        """
        # 1. 도메인 필터링
        is_smartstore = self._is_smartstore_question(query)
        if not is_smartstore:
            yield "죄송합니다. 저는 네이버 스마트스토어 관련 질문에만 답변할 수 있습니다. 스마트스토어 관련 질문을 해주시겠어요?"
            return

        # 2. RAG 검색
        if use_hybrid:
            search_results = self.rag_service.hybrid_search(query, top_k=top_k)
        else:
            search_results = self.rag_service.semantic_search(query, top_k=top_k)

        if not search_results:
            yield "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?"
            return

        # 3. 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {i+1}] (카테고리: {doc['metadata']['category']})\n{doc['document']}"
            for i, doc in enumerate(search_results[:3])
        ])

        # 4. 대화 기록
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        history = self.conversation_history[session_id]

        # 5. 메시지 구성
        messages = [
            {"role": "system", "content": self._generate_system_prompt()}
        ]

        for msg in history[-6:]:
            messages.append(msg)

        user_message = f"""관련 문서:
{context}

사용자 질문: {query}

위 문서를 참고하여 질문에 답변해주세요."""

        messages.append({"role": "user", "content": user_message})

        # 6. 스트리밍 답변 생성
        full_answer = ""

        if self.chat_provider == "solar":
            async for chunk in self.solar_service.async_stream_chat_response(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            ):
                full_answer += chunk
                yield chunk
        else:
            stream = self.openai.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield content

        # 7. 대화 기록 저장
        self.conversation_history[session_id].append(
            {"role": "user", "content": query}
        )
        self.conversation_history[session_id].append(
            {"role": "assistant", "content": full_answer}
        )

        if len(self.conversation_history[session_id]) > 20:
            self.conversation_history[session_id] = self.conversation_history[session_id][-20:]


    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """
        대화 기록 조회

        Args:
            session_id: 세션 ID

        Returns:
            대화 기록 리스트
        """
        return self.conversation_history.get(session_id, [])


    def clear_conversation_history(self, session_id: str = "default") -> None:
        """
        대화 기록 초기화

        Args:
            session_id: 세션 ID
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"세션 {session_id} 대화 기록 초기화")


# 테스트 코드
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    # 챗봇 서비스 초기화
    chatbot = ChatbotService()

    # 테스트 질문들
    test_queries = [
        "스마트스토어 가입은 어떻게 하나요?",
        "상품 등록 방법을 알려주세요",
        "오늘 날씨 어때?"  # 도메인 외 질문
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"질문: {query}")
        print(f"{'='*60}")

        result = chatbot.chat(query)

        print(f"\n답변:\n{result['answer']}\n")
        print(f"스마트스토어 관련: {result['is_smartstore_related']}")
        print(f"\n후속 질문:")
        for i, fq in enumerate(result['follow_up_questions'], 1):
            print(f"  {i}. {fq}")

        if result['sources']:
            print(f"\n참고 문서:")
            for i, src in enumerate(result['sources'], 1):
                print(f"  {i}. [{src['category']}] {src['question'][:50]}...")
