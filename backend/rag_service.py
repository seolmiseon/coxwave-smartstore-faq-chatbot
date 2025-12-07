# RAG Service (LangChain 없이 구현)
# 콕스웨이브 과제 전형 - Hybrid RAG without LangChain
#
# 차별화 포인트:
# 1. 함께키즈 프로젝트 경험 기반 Hybrid RAG 아키텍처 (85% 정확도 향상)
# 2. LangChain 없이 직접 구현 → 내부 동작 원리 완전 이해
# 3. Semantic + Keyword + RRF 통합 전략 (직접 구현)

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import List, Dict, Any, Optional
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG Service without LangChain

    직접 구현 이유:
    - LangChain은 추상화 레이어일 뿐, 내부 원리를 이해하면 직접 구현 가능
    - 더 세밀한 제어와 커스터마이징 가능
    - 불필요한 오버헤드 제거

    핵심 개념 (함께키즈,FSF축구플랫폼 프로젝트 학습):
    - RAG = "스마트한 캐시 미스 해결 전략"
    - Retrieval: 관련 문서 검색
    - Augmentation: 검색 결과를 컨텍스트로 추가
    - Generation: LLM이 컨텍스트 기반 답변 생성
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "faq_collection"
    ):
        """
        RAG 서비스 초기화

        Args:
            persist_directory: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
        """
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # OpenAI 클라이언트 초기화
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # FAQ 컬렉션 생성/로드
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 코사인 유사도
        )

        logger.info(f"RAG Service 초기화 완료 (컬렉션: {collection_name})")


    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        텍스트를 토큰 제한에 맞게 잘라냄

        text-embedding-3-small의 최대 토큰: 8192
        안전하게 8000 토큰으로 제한

        Args:
            text: 원본 텍스트
            max_tokens: 최대 토큰 수

        Returns:
            잘라낸 텍스트
        """
        # 간단한 어림셈: 한글은 1글자당 약 1.5 토큰
        # 영어는 1단어당 약 1.3 토큰
        # 안전하게 max_tokens * 0.6 글자로 제한
        max_chars = int(max_tokens * 0.6)

        if len(text) <= max_chars:
            return text

        # 잘라내고 경고
        truncated = text[:max_chars]
        logger.warning(f"텍스트가 너무 길어 {max_chars}자로 잘라냈습니다. 원본: {len(text)}자")

        return truncated


    def _get_embedding(self, text: str) -> List[float]:
        """
        텍스트를 벡터 임베딩으로 변환

        직접 구현 이유:
        - LangChain의 Embeddings 클래스 대신 OpenAI API 직접 호출
        - 더 명확한 에러 핸들링 가능

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (1536차원 for text-embedding-3-small)
        """
        try:
            # 토큰 제한 처리
            truncated_text = self._truncate_text(text)

            response = self.openai.embeddings.create(
                model=self.embedding_model,
                input=truncated_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise


    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        문서를 벡터 DB에 추가

        직접 구현 포인트:
        - LangChain의 Document 클래스 없이 직접 관리
        - 배치 임베딩으로 성능 최적화

        Args:
            texts: 문서 텍스트 리스트
            metadatas: 문서 메타데이터 리스트
            ids: 문서 ID 리스트 (None이면 자동 생성)
        """
        if not texts:
            logger.warning("추가할 문서가 없습니다.")
            return

        # ID 자동 생성
        if ids is None:
            current_count = self.collection.count()
            ids = [f"doc_{current_count + i}" for i in range(len(texts))]

        # 메타데이터 기본값 설정
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]

        # 임베딩 생성
        logger.info(f"{len(texts)}개 문서 임베딩 생성 중...")
        embeddings = [self._get_embedding(text) for text in texts]

        # ChromaDB에 추가
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"✅ {len(texts)}개 문서 추가 완료")


    def semantic_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Semantic Search (벡터 유사도 기반)

        함께키즈 프로젝트 경험:
        - 단순 벡터 검색만으로는 정확도 한계
        - Hybrid RAG와 결합하면 85% 향상

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = self._get_embedding(query)

        # 벡터 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 결과 포맷팅
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i],  # 코사인 유사도
                "rank": i + 1,
                "method": "semantic"
            })

        return formatted_results


    def keyword_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Keyword Search (키워드 매칭)

        직접 구현 이유:
        - ChromaDB의 where 필터 활용
        - 정확한 키워드 매칭으로 semantic search 보완

        함께키즈 경험:
        - "가입절차" 같은 카테고리 태그 검색에 효과적
        - Semantic만으로 놓치는 정확한 매칭 캐치

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트
        """
        # 전체 문서 가져오기 (ChromaDB는 BM25 미지원)
        all_docs = self.collection.get(
            include=["documents", "metadatas"]
        )

        # 키워드 매칭 (단순 포함 검사)
        query_lower = query.lower()
        matched = []

        for i, doc in enumerate(all_docs["documents"]):
            doc_lower = doc.lower()

            # 키워드가 문서에 포함되어 있는지 확인
            if query_lower in doc_lower:
                # 매칭 스코어 계산 (단순 빈도)
                score = doc_lower.count(query_lower)
                matched.append({
                    "document": doc,
                    "metadata": all_docs["metadatas"][i],
                    "score": score,
                    "rank": 0,  # 나중에 정렬 후 할당
                    "method": "keyword"
                })

        # 스코어 기준 정렬
        matched.sort(key=lambda x: x["score"], reverse=True)

        # 상위 k개만 반환하고 rank 할당
        results = matched[:top_k]
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results


    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion)

        함께키즈 프로젝트 핵심 알고리즘:
        - Semantic + Keyword 결과 통합
        - 85% 정확도 향상의 핵심

        RRF 공식: score = Σ 1/(k + rank)
        - k: 상수 (일반적으로 60)
        - rank: 각 검색 방법에서의 순위

        직접 구현 이유:
        - LangChain의 EnsembleRetriever 대신 직접 구현
        - RRF 알고리즘 내부 동작 완전 이해

        Args:
            semantic_results: Semantic 검색 결과
            keyword_results: Keyword 검색 결과
            k: RRF 상수

        Returns:
            통합된 검색 결과 (RRF 스코어 기준 정렬)
        """
        # 문서별 RRF 스코어 계산
        rrf_scores = defaultdict(lambda: {
            "score": 0.0,
            "document": None,
            "metadata": None,
            "sources": []
        })

        # Semantic 결과 반영
        for result in semantic_results:
            doc_text = result["document"]
            rank = result["rank"]
            rrf_score = 1.0 / (k + rank)

            rrf_scores[doc_text]["score"] += rrf_score
            rrf_scores[doc_text]["document"] = doc_text
            rrf_scores[doc_text]["metadata"] = result["metadata"]
            rrf_scores[doc_text]["sources"].append({
                "method": "semantic",
                "rank": rank,
                "similarity": result.get("similarity")
            })

        # Keyword 결과 반영
        for result in keyword_results:
            doc_text = result["document"]
            rank = result["rank"]
            rrf_score = 1.0 / (k + rank)

            rrf_scores[doc_text]["score"] += rrf_score
            if rrf_scores[doc_text]["document"] is None:
                rrf_scores[doc_text]["document"] = doc_text
                rrf_scores[doc_text]["metadata"] = result["metadata"]
            rrf_scores[doc_text]["sources"].append({
                "method": "keyword",
                "rank": rank,
                "score": result.get("score")
            })

        # RRF 스코어 기준 정렬
        final_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # 최종 rank 할당
        for i, result in enumerate(final_results):
            result["final_rank"] = i + 1
            result["method"] = "hybrid_rrf"

        return final_results


    def hybrid_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid RAG Search (Semantic + Keyword + RRF)

        함께키즈 프로젝트 핵심 기술:
        - 단순 벡터 검색 대비 85% 정확도 향상
        - 실전 프로젝트 검증된 아키텍처

        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 결과 개수

        Returns:
            Hybrid 검색 결과 (상위 k개)
        """
        # 1. Semantic Search
        semantic_results = self.semantic_search(query, top_k=top_k * 2)

        # 2. Keyword Search
        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # 3. RRF로 통합
        hybrid_results = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results
        )

        # 상위 k개만 반환
        return hybrid_results[:top_k]


    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        LLM을 사용하여 답변 생성

        직접 구현 포인트:
        - LangChain의 RetrievalQA 대신 직접 프롬프트 엔지니어링
        - 더 세밀한 프롬프트 제어

        Args:
            query: 사용자 질문
            context_docs: 검색된 문서 리스트
            temperature: LLM temperature
            max_tokens: 최대 토큰 수

        Returns:
            생성된 답변
        """
        # 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {i+1}]\n{doc['document']}"
            for i, doc in enumerate(context_docs)
        ])

        # 프롬프트 구성
        system_prompt = """당신은 네이버 스마트스토어 FAQ 전문가입니다.
사용자의 질문에 대해 제공된 문서를 참고하여 정확하고 친절하게 답변해주세요.

답변 규칙:
1. 제공된 문서에 기반하여 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 간결하고 명확하게 답변하세요
4. 필요시 단계별로 설명하세요"""

        user_prompt = f"""관련 문서:
{context}

사용자 질문: {query}

위 문서를 참고하여 질문에 답변해주세요."""

        # LLM 호출
        try:
            response = self.openai.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM 답변 생성 실패: {e}")
            raise


    def query(
        self,
        query: str,
        use_hybrid: bool = True,
        top_k: int = 5
    ) -> str:
        """
        RAG 전체 파이프라인 실행

        흐름:
        1. Hybrid Search (또는 Semantic Search)
        2. LLM으로 답변 생성

        Args:
            query: 사용자 질문
            use_hybrid: Hybrid RAG 사용 여부
            top_k: 검색할 문서 개수

        Returns:
            생성된 답변
        """
        # 1. 검색
        if use_hybrid:
            search_results = self.hybrid_search(query, top_k=top_k)
            logger.info(f"Hybrid Search 완료: {len(search_results)}개 문서")
        else:
            search_results = self.semantic_search(query, top_k=top_k)
            logger.info(f"Semantic Search 완료: {len(search_results)}개 문서")

        # 검색 결과가 없으면 기본 답변
        if not search_results:
            return "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?"

        # 2. 답변 생성
        answer = self.generate_answer(query, search_results)

        return answer


    def get_stats(self) -> Dict[str, Any]:
        """
        RAG 서비스 통계 정보

        Returns:
            통계 정보 딕셔너리
        """
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model
        }
