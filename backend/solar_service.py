# Upstage Solar Pro 서비스
# OpenAI SDK 재사용 (base_url만 변경)

from openai import OpenAI
import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

logger = logging.getLogger(__name__)


class SolarService:
    """
    Upstage Solar Pro 서비스

    차별화 포인트:
    - OpenAI SDK 재사용 (base_url만 변경하면 됨!)
    - 한국어 특화 모델 (네이버 스마트스토어 FAQ에 최적)
    - solar-pro2 사용
    """

    def __init__(self):
        """Solar 클라이언트 초기화"""
        api_key = os.getenv("SOLAR_API_KEY")
        if not api_key:
            raise ValueError("SOLAR_API_KEY 환경변수가 설정되지 않았습니다.")

        # OpenAI SDK 재사용 - base_url만 변경!
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("SOLAR_BASE_URL", "https://api.upstage.ai/v1")
        )

        self.chat_model = os.getenv("SOLAR_MODEL", "solar-pro2")
        self.embedding_model = os.getenv("SOLAR_EMBEDDING_MODEL", "solar-embedding-1-large")

        logger.info(f"Solar Service 초기화 완료 (모델: {self.chat_model})")


    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Solar Pro로 채팅 응답 생성

        Args:
            messages: 메시지 리스트 [{"role": "user", "content": "..."}]
            temperature: 온도 (0.0 ~ 1.0)
            max_tokens: 최대 토큰 수

        Returns:
            생성된 응답 텍스트
        """
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Solar 응답 생성 실패: {e}")
            raise


    def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Solar Pro로 스트리밍 채팅 응답 생성

        Args:
            messages: 메시지 리스트
            temperature: 온도
            max_tokens: 최대 토큰 수

        Yields:
            응답 청크 (문자열)
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Solar 스트리밍 응답 실패: {e}")
            raise


    async def async_stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncIterator[str]:
        """
        Solar Pro로 비동기 스트리밍 채팅 응답 생성

        FastAPI와 함께 사용

        Args:
            messages: 메시지 리스트
            temperature: 온도
            max_tokens: 최대 토큰 수

        Yields:
            응답 청크 (문자열)
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Solar 비동기 스트리밍 응답 실패: {e}")
            raise


    def get_embedding(self, text: str) -> List[float]:
        """
        Solar Embedding으로 벡터 생성

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Solar 임베딩 생성 실패: {e}")
            raise


# 테스트 코드
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    # Solar 서비스 초기화
    solar = SolarService()

    # 테스트 메시지
    messages = [
        {
            "role": "user",
            "content": "스마트스토어 가입은 어떻게 하나요?"
        }
    ]

    # 일반 응답 테스트
    print("=== 일반 응답 테스트 ===")
    response = solar.generate_chat_response(messages)
    print(f"응답: {response}\n")

    # 스트리밍 응답 테스트
    print("=== 스트리밍 응답 테스트 ===")
    for chunk in solar.stream_chat_response(messages):
        print(chunk, end="", flush=True)
    print("\n")

    # 임베딩 테스트
    print("=== 임베딩 테스트 ===")
    embedding = solar.get_embedding("테스트 텍스트")
    print(f"임베딩 차원: {len(embedding)}")
    print(f"첫 5개 값: {embedding[:5]}")
