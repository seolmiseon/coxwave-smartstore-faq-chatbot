# FAQ 데이터 ChromaDB 임베딩 스크립트
# 콕스웨이브 과제 전형 - 2717개 FAQ 로딩

import pickle
import logging
from pathlib import Path
from dotenv import load_dotenv
from rag_service import RAGService

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_faq_from_pickle(pkl_path: str) -> dict:
    """
    final_result.pkl 파일 로드

    데이터 구조:
    {
        "[카테고리] 질문": "답변",
        ...
    }

    Args:
        pkl_path: pickle 파일 경로

    Returns:
        FAQ 딕셔너리
    """
    logger.info(f"Loading FAQ data from {pkl_path}...")

    with open(pkl_path, "rb") as f:
        faq_data = pickle.load(f)

    logger.info(f"✅ Loaded {len(faq_data)} FAQ entries")
    return faq_data


def extract_category_from_question(question: str) -> str:
    """
    질문에서 카테고리 추출

    예시:
    - "[가입절차] 질문내용" → "가입절차"
    - "일반 질문" → "기타"

    Args:
        question: 질문 텍스트

    Returns:
        카테고리 문자열
    """
    if question.startswith("[") and "]" in question:
        category = question.split("]")[0].strip("[")
        return category
    return "기타"


def prepare_documents_for_embedding(faq_data: dict):
    """
    FAQ 데이터를 임베딩용 형식으로 변환

    변환 전략:
    - 질문과 답변을 함께 임베딩 (검색 정확도 향상)
    - 메타데이터에 카테고리, 원본 질문 저장

    Args:
        faq_data: FAQ 딕셔너리

    Returns:
        (texts, metadatas, ids) 튜플
    """
    texts = []
    metadatas = []
    ids = []

    for idx, (question, answer) in enumerate(faq_data.items()):
        # 카테고리 추출
        category = extract_category_from_question(question)

        # 질문에서 카테고리 제거 (클린한 질문 텍스트)
        clean_question = question.split("]")[-1].strip() if "]" in question else question

        # 임베딩할 텍스트 구성
        # 전략: "질문\n답변" 형태로 저장
        # 이유: 질문과 답변 모두 검색 대상 (함께키즈 경험)
        combined_text = f"질문: {clean_question}\n답변: {answer}"

        texts.append(combined_text)
        metadatas.append({
            "category": category,
            "original_question": question,
            "clean_question": clean_question,
            "answer": answer,
            "source": "smartstore_faq"
        })
        ids.append(f"faq_{idx}")

    logger.info(f"Prepared {len(texts)} documents for embedding")

    # 카테고리 분포 출력
    category_counts = {}
    for meta in metadatas:
        cat = meta["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat}: {count}")

    return texts, metadatas, ids


def main():
    """
    메인 실행 함수

    흐름:
    1. final_result.pkl 로드
    2. 임베딩용 형식으로 변환
    3. RAGService를 사용하여 ChromaDB에 저장
    """
    logger.info("=== FAQ 데이터 임베딩 시작 ===")

    # 1. PKL 파일 로드
    pkl_path = Path(__file__).parent.parent / "final_result.pkl"
    if not pkl_path.exists():
        logger.error(f"PKL 파일을 찾을 수 없습니다: {pkl_path}")
        return

    faq_data = load_faq_from_pickle(str(pkl_path))

    # 2. 임베딩용 데이터 준비
    texts, metadatas, ids = prepare_documents_for_embedding(faq_data)

    # 3. RAG 서비스 초기화
    rag_service = RAGService(
        persist_directory="./chroma_db",
        collection_name="smartstore_faq"
    )

    # 4. 기존 데이터 확인
    stats = rag_service.get_stats()
    if stats["total_documents"] > 0:
        logger.warning(f"기존 데이터 {stats['total_documents']}개 발견!")
        response = input("기존 데이터를 삭제하고 다시 로드하시겠습니까? (y/N): ")
        if response.lower() == "y":
            # 컬렉션 삭제 후 재생성
            rag_service.client.delete_collection("smartstore_faq")
            logger.info("기존 컬렉션 삭제 완료")
            rag_service = RAGService(
                persist_directory="./chroma_db",
                collection_name="smartstore_faq"
            )
        else:
            logger.info("로딩 취소")
            return

    # 5. 문서 추가 (배치 처리)
    logger.info("문서 임베딩 및 저장 시작... (시간이 다소 걸릴 수 있습니다)")

    # 배치 크기 설정 (OpenAI API 속도 제한 고려)
    batch_size = 100
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        logger.info(f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch_texts)}개 문서)")

        rag_service.add_documents(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    # 6. 최종 통계
    final_stats = rag_service.get_stats()
    logger.info("=== 임베딩 완료 ===")
    logger.info(f"총 문서 수: {final_stats['total_documents']}")
    logger.info(f"컬렉션: {final_stats['collection_name']}")
    logger.info(f"임베딩 모델: {final_stats['embedding_model']}")
    logger.info(f"채팅 모델: {final_stats['chat_model']}")

    # 7. 테스트 검색
    logger.info("\n=== 테스트 검색 ===")
    test_query = "스마트스토어 가입은 어떻게 하나요?"
    logger.info(f"테스트 쿼리: {test_query}")

    # Hybrid Search 테스트
    results = rag_service.hybrid_search(test_query, top_k=3)
    logger.info(f"검색 결과 {len(results)}개:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n[{i}] (RRF Score: {result['score']:.4f})")
        logger.info(f"카테고리: {result['metadata']['category']}")
        logger.info(f"질문: {result['metadata']['clean_question'][:100]}...")


if __name__ == "__main__":
    main()
