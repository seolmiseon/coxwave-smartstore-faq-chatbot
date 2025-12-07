# 글로벌 쿼리 캐싱 서비스 (FSF 축구 플랫폼 전략 적용)
# 콕스웨이브 과제 전형

from openai import OpenAI
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib
import chromadb

logger = logging.getLogger(__name__)


class QueryCacheService:
    """
    글로벌 쿼리 캐싱 서비스 (FSF 축구 플랫폼 전략 적용)

    핵심 기능:
    1. 유사 질문 검색 (임베딩 기반)
    2. 캐시 히트 시 LLM 호출 생략 → 90% 비용 절감
    3. 새 응답 자동 캐싱

    사용 사례:
        User A: "스마트스토어 가입 방법 알려줘"
        User B: "스마트스토어 가입은 어떻게 하나요?"
        User C: "스마트스토어 신청 절차는?"
        → 유사도 90% 이상이면 같은 답변 재사용 (LLM 호출 1번만!)
    """

    def __init__(
        self,
        cache_path: str = "./data/chroma_cache",
        similarity_threshold: float = 0.90
    ):
        """
        쿼리 캐시 초기화

        Args:
            cache_path: ChromaDB 저장 경로
            similarity_threshold: 유사도 임계값 (0.90 = 90% 이상 유사)
        """
        self.client = chromadb.PersistentClient(path=cache_path)
        self.cache_collection = self.client.get_or_create_collection(
            name="query_cache",
            metadata={"hnsw:space": "cosine"}  # Cosine 유사도
        )
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.threshold = similarity_threshold

        # 통계
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"QueryCacheService 초기화 완료 (threshold: {similarity_threshold})")

    def _get_cache_key(self, query: str) -> str:
        """질문을 해시값으로 변환 (정확히 같은 질문 판별)"""
        return hashlib.md5(query.encode()).hexdigest()

    def search_similar_cache(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """
        유사한 캐시된 답변 검색

        Args:
            query: 사용자 질문
            threshold: 유사도 임계값 (기본값: 0.90)

        Returns:
            캐시 히트 시:
                {
                    "cached": True,
                    "answer": "답변 내용",
                    "similarity": 0.95,
                    "original_query": "원래 질문",
                    "timestamp": "2025-12-07T...",
                    "follow_up_questions": [...],
                    "sources": [...]
                }
            캐시 미스 시: None
        """
        try:
            if threshold is None:
                threshold = self.threshold

            # 1. 질문을 벡터로 변환
            embedding = (
                self.openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=query
                )
                .data[0]
                .embedding
            )

            # 2. ChromaDB에서 유사 질문 검색
            results = self.cache_collection.query(
                query_embeddings=[embedding],
                n_results=1
            )

            # 3. 유사도 체크
            if results["distances"][0]:
                distance = results["distances"][0][0]
                similarity = 1 - distance  # cosine distance → similarity

                if similarity >= threshold:
                    self.cache_hits += 1
                    metadata = results["metadatas"][0][0]

                    logger.info(
                        f"🎯 캐시 HIT! 유사도: {similarity:.2%} "
                        f"(원래 질문: {metadata['query']})"
                    )

                    return {
                        "cached": True,
                        "answer": metadata["answer"],
                        "similarity": similarity,
                        "original_query": metadata["query"],
                        "timestamp": metadata.get("timestamp", ""),
                        "follow_up_questions": json.loads(metadata.get("follow_ups", "[]")),
                        "sources": json.loads(metadata.get("sources", "[]"))
                    }
                else:
                    self.cache_misses += 1
                    logger.info(f"캐시 MISS (유사도 {similarity:.2%} < {threshold:.2%})")

            return None

        except Exception as e:
            logger.error(f"캐시 검색 실패: {e}")
            return None

    def save_cache(
        self,
        query: str,
        answer: str,
        follow_up_questions: List[str],
        sources: List[Dict[str, Any]]
    ):
        """
        새 답변을 캐시에 저장

        Args:
            query: 사용자 질문
            answer: LLM 응답
            follow_up_questions: 후속 질문 리스트
            sources: 참고 문서 리스트
        """
        try:
            # 1. 질문을 벡터로 변환
            embedding = (
                self.openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=query
                )
                .data[0]
                .embedding
            )

            # 2. ChromaDB에 저장
            cache_key = self._get_cache_key(query)

            self.cache_collection.upsert(
                ids=[cache_key],
                embeddings=[embedding],
                documents=[query],  # 검색용 (실제론 사용 안함)
                metadatas=[{
                    "query": query,
                    "answer": answer,
                    "follow_ups": json.dumps(follow_up_questions, ensure_ascii=False),
                    "sources": json.dumps(sources, ensure_ascii=False),
                    "timestamp": datetime.now().isoformat()
                }]
            )

            logger.info(f"💾 캐시 저장 완료: {query[:50]}...")

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")

    def get_cache_stats(self) -> Dict:
        """
        캐시 통계 조회

        Returns:
            {
                "total_cached": 100,
                "cache_hits": 50,
                "cache_misses": 10,
                "hit_rate": 0.833,
                "cache_path": "./data/chroma_cache"
            }
        """
        try:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

            return {
                "total_cached": self.cache_collection.count(),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": round(hit_rate, 3)
            }

        except Exception as e:
            logger.error(f"캐시 통계 조회 실패: {e}")
            return {
                "total_cached": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0.0
            }
