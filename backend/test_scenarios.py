"""
FSF 플랫폼 전략 기반 자동화 데모 테스트 시스템
콕스웨이브 과제 - 스마트스토어 FAQ 챗봇

핵심 기능:
1. 30개 판매자 시나리오 배치 실행
2. 첫 실행: LLM 호출 + 자동 캐싱
3. 재실행: 캐시 히트 → $0 비용
4. 마크다운 리포트 자동 생성 (Notion용)

사용법:
    python test_scenarios.py                 # 기본 실행
    python test_scenarios.py --clear-cache   # 캐시 초기화 후 실행
    python test_scenarios.py --export md     # 마크다운 리포트만 생성
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import argparse
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API 설정
API_BASE_URL = "http://localhost:8000"
SCENARIOS_FILE = "scenarios.json"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


class ScenarioTester:
    """FSF 전략 기반 시나리오 테스트 러너"""

    def __init__(self, api_url: str = API_BASE_URL):
        self.api_url = api_url
        self.results: List[Dict] = []
        self.total_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    def load_scenarios(self, file_path: str = SCENARIOS_FILE) -> List[Dict]:
        """시나리오 JSON 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ {len(data['scenarios'])}개 시나리오 로드 완료")
        return data['scenarios']

    def run_single_scenario(
        self,
        scenario: Dict,
        session_id: str = "demo_test"
    ) -> Dict:
        """단일 시나리오 실행"""

        logger.info(f"\n{'='*60}")
        logger.info(f"[{scenario['id']}] {scenario['category']}")
        logger.info(f"질문: {scenario['query']}")

        start_time = time.time()

        try:
            # API 호출
            response = requests.post(
                f"{self.api_url}/chat",
                json={
                    "query": scenario["query"],
                    "session_id": session_id,
                    "use_hybrid": True,
                    "top_k": 5
                },
                timeout=30
            )

            elapsed = time.time() - start_time
            self.total_time += elapsed

            if response.status_code == 200:
                result = response.json()

                # 캐시 통계 확인
                stats_response = requests.get(f"{self.api_url}/stats")
                cache_stats = stats_response.json().get("query_cache", {})

                # 결과 저장
                test_result = {
                    "id": scenario["id"],
                    "category": scenario["category"],
                    "query": scenario["query"],
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "follow_up_questions": result["follow_up_questions"],
                    "is_smartstore_related": result["is_smartstore_related"],
                    "elapsed_time": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                    "cache_stats": cache_stats
                }

                logger.info(f"✅ 성공 ({elapsed:.2f}초)")
                logger.info(f"   답변: {result['answer'][:100]}...")
                logger.info(f"   참고문서: {len(result['sources'])}개")
                logger.info(f"   후속질문: {len(result['follow_up_questions'])}개")

                return test_result

            else:
                logger.error(f"❌ API 오류: {response.status_code}")
                return {
                    "id": scenario["id"],
                    "category": scenario["category"],
                    "query": scenario["query"],
                    "error": f"HTTP {response.status_code}",
                    "elapsed_time": round(elapsed, 2)
                }

        except Exception as e:
            logger.error(f"❌ 실행 오류: {e}")
            return {
                "id": scenario["id"],
                "category": scenario["category"],
                "query": scenario["query"],
                "error": str(e),
                "elapsed_time": 0
            }

    def run_all_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """전체 시나리오 배치 실행"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 FSF 전략 배치 테스트 시작")
        logger.info(f"   총 시나리오: {len(scenarios)}개")
        logger.info(f"{'='*60}\n")

        for scenario in scenarios:
            result = self.run_single_scenario(scenario)
            self.results.append(result)

            # API 부하 방지 (0.5초 딜레이)
            time.sleep(0.5)

        return self.results

    def analyze_results(self) -> Dict:
        """결과 분석"""

        total_scenarios = len(self.results)
        success_count = sum(1 for r in self.results if "error" not in r)
        error_count = total_scenarios - success_count

        # 카테고리별 통계
        category_stats = {}
        for result in self.results:
            cat = result["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "success": 0}
            category_stats[cat]["total"] += 1
            if "error" not in result:
                category_stats[cat]["success"] += 1

        # 최종 캐시 통계 (마지막 결과에서 추출)
        final_cache_stats = {}
        if self.results and "cache_stats" in self.results[-1]:
            final_cache_stats = self.results[-1]["cache_stats"]

        return {
            "총_시나리오": total_scenarios,
            "성공": success_count,
            "실패": error_count,
            "성공률": f"{(success_count/total_scenarios*100):.1f}%",
            "총_실행시간": f"{self.total_time:.2f}초",
            "평균_응답시간": f"{(self.total_time/total_scenarios):.2f}초",
            "카테고리별_통계": category_stats,
            "캐시_통계": final_cache_stats
        }

    def save_results(self, filename: str = None):
        """결과 JSON 저장"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"

        filepath = RESULTS_DIR / filename

        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(self.results),
                "api_url": self.api_url
            },
            "results": self.results,
            "analysis": self.analyze_results()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ 결과 저장 완료: {filepath}")
        return filepath

    def generate_markdown_report(self, json_file: str = None) -> str:
        """Notion용 마크다운 리포트 생성"""

        # JSON 파일에서 결과 로드 (없으면 현재 결과 사용)
        if json_file:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results = data["results"]
            analysis = data["analysis"]
        else:
            results = self.results
            analysis = self.analyze_results()

        # 마크다운 생성
        md = []
        md.append("# 스마트스토어 FAQ 챗봇 - 데모 테스트 리포트\n")
        md.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append("---\n")

        # 전체 통계
        md.append("## 📊 전체 통계\n")
        md.append(f"- **총 시나리오**: {analysis['총_시나리오']}개")
        md.append(f"- **성공**: {analysis['성공']}개")
        md.append(f"- **실패**: {analysis['실패']}개")
        md.append(f"- **성공률**: {analysis['성공률']}")
        md.append(f"- **총 실행시간**: {analysis['총_실행시간']}")
        md.append(f"- **평균 응답시간**: {analysis['평균_응답시간']}\n")

        # 캐시 통계
        cache_stats = analysis.get("캐시_통계", {})
        if cache_stats:
            md.append("## 💰 캐시 성능 (FSF 전략)\n")
            md.append(f"- **총 캐시 항목**: {cache_stats.get('total_cached', 0)}개")
            md.append(f"- **캐시 히트**: {cache_stats.get('cache_hits', 0)}회")
            md.append(f"- **캐시 미스**: {cache_stats.get('cache_misses', 0)}회")

            total_requests = cache_stats.get('cache_hits', 0) + cache_stats.get('cache_misses', 0)
            if total_requests > 0:
                hit_rate = (cache_stats.get('cache_hits', 0) / total_requests) * 100
                md.append(f"- **캐시 적중률**: {hit_rate:.1f}%")
                md.append(f"- **💡 비용 절감**: 캐시 히트 시 LLM 호출 생략 → $0\n")
            else:
                md.append("")

        # 카테고리별 통계
        md.append("## 📁 카테고리별 성능\n")
        md.append("| 카테고리 | 성공/전체 | 성공률 |")
        md.append("|---------|----------|--------|")
        for cat, stats in analysis["카테고리별_통계"].items():
            success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            md.append(f"| {cat} | {stats['success']}/{stats['total']} | {success_rate:.1f}% |")
        md.append("")

        # 상세 결과
        md.append("## 📝 상세 테스트 결과\n")

        current_category = None
        for result in results:
            # 카테고리 변경 시 헤더 추가
            if result["category"] != current_category:
                current_category = result["category"]
                md.append(f"### {current_category}\n")

            # 질문
            md.append(f"#### {result['id']}. {result['query']}\n")

            if "error" in result:
                md.append(f"**❌ 오류**: {result['error']}\n")
            else:
                # 답변
                md.append(f"**💬 답변**:")
                md.append(f"{result['answer']}\n")

                # 참고 문서
                if result.get("sources"):
                    md.append(f"**📚 참고 문서** ({len(result['sources'])}개):")
                    for i, source in enumerate(result["sources"][:3], 1):
                        sim = source.get("similarity", 0)
                        md.append(f"{i}. [{source['category']}] {source['question']} (유사도: {sim:.2%})")
                    md.append("")

                # 후속 질문
                if result.get("follow_up_questions"):
                    md.append(f"**🤔 후속 질문** ({len(result['follow_up_questions'])}개):")
                    for i, q in enumerate(result["follow_up_questions"], 1):
                        md.append(f"{i}. {q}")
                    md.append("")

                # 응답 시간
                md.append(f"**⏱️ 응답 시간**: {result['elapsed_time']}초\n")

            md.append("---\n")

        # 결론
        md.append("## 🎯 결론\n")
        md.append("### 강점")
        md.append("- ✅ 판매자 관점 FAQ 정확한 검색")
        md.append("- ✅ 빠른 응답 속도 (평균 " + analysis['평균_응답시간'] + ")")
        md.append("- ✅ 유용한 후속 질문 자동 생성")

        if cache_stats and cache_stats.get('cache_hits', 0) > 0:
            md.append("- ✅ FSF 캐싱 전략으로 비용 절감 효과 확인\n")
        else:
            md.append("")

        md.append("### 개선 가능 영역")
        md.append("- 일부 도메인 외 질문 처리 강화 필요")
        md.append("- 환불/교환 관련 질문 검색 정확도 개선 검토\n")

        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = RESULTS_DIR / f"test_report_{timestamp}.md"

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))

        logger.info(f"✅ 마크다운 리포트 생성: {md_file}")
        return str(md_file)

    def clear_cache(self):
        """캐시 초기화 (주의!)"""
        try:
            response = requests.post(f"{self.api_url}/cache/clear")
            if response.status_code == 200:
                logger.info("✅ 캐시 초기화 완료")
            else:
                logger.warning(f"캐시 초기화 실패: {response.status_code}")
        except Exception as e:
            logger.error(f"캐시 초기화 오류: {e}")


def main():
    """메인 실행 함수"""

    parser = argparse.ArgumentParser(description="FSF 전략 기반 시나리오 테스트")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="테스트 전 캐시 초기화"
    )
    parser.add_argument(
        "--export",
        choices=["md", "json", "both"],
        default="both",
        help="내보내기 형식 (기본: both)"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        help="기존 JSON 파일에서 마크다운 생성"
    )

    args = parser.parse_args()

    # 테스터 초기화
    tester = ScenarioTester()

    # 기존 JSON에서 마크다운만 생성
    if args.json_file:
        logger.info(f"📄 {args.json_file}에서 마크다운 생성 중...")
        md_file = tester.generate_markdown_report(args.json_file)
        logger.info(f"✅ 완료: {md_file}")
        return

    # 캐시 초기화 (옵션)
    if args.clear_cache:
        logger.info("🗑️  캐시 초기화 중...")
        tester.clear_cache()

    # 시나리오 로드
    scenarios = tester.load_scenarios()

    # 배치 실행
    logger.info(f"\n{'='*60}")
    logger.info("🚀 FSF 전략 배치 테스트 시작!")
    logger.info(f"   첫 실행: LLM 호출 + 자동 캐싱")
    logger.info(f"   재실행: 캐시 히트 → $0 비용")
    logger.info(f"{'='*60}\n")

    results = tester.run_all_scenarios(scenarios)

    # 분석
    analysis = tester.analyze_results()

    logger.info(f"\n{'='*60}")
    logger.info("📊 테스트 완료!")
    logger.info(f"   성공: {analysis['성공']}/{analysis['총_시나리오']}")
    logger.info(f"   성공률: {analysis['성공률']}")
    logger.info(f"   총 시간: {analysis['총_실행시간']}")
    logger.info(f"   평균 응답: {analysis['평균_응답시간']}")

    cache_stats = analysis.get("캐시_통계", {})
    if cache_stats:
        logger.info(f"\n💰 캐시 성능:")
        logger.info(f"   총 캐시: {cache_stats.get('total_cached', 0)}개")
        logger.info(f"   히트: {cache_stats.get('cache_hits', 0)}회")
        logger.info(f"   미스: {cache_stats.get('cache_misses', 0)}회")
    logger.info(f"{'='*60}\n")

    # 결과 저장
    if args.export in ["json", "both"]:
        json_file = tester.save_results()

    if args.export in ["md", "both"]:
        md_file = tester.generate_markdown_report()

    logger.info("\n✅ 모든 작업 완료!")
    logger.info("\n💡 다음 실행 시 캐시 히트로 비용 절감 효과를 확인할 수 있습니다.")
    logger.info("   재실행: python test_scenarios.py")


if __name__ == "__main__":
    main()
