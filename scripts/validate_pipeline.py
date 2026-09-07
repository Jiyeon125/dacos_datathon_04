from __future__ import annotations

import argparse
import math

from src.accessibility_simulator import simulate_accessibility_for_candidates
from src.data_loader import load_bundle
from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.resource_benchmark import build_resource_scenario_table
from src.scenario_ranking import build_priority_ranking
from src.scenario_engine import run_scenario
from src.schema import KEDI, SCHOOL_NAME, STUDENTS
from src.validation import assert_valid_bundle


KNOWN_A = "213021106"  # 가남초
KNOWN_B = "213021124"  # 가야초


def assert_close(name: str, actual: float, expected: float, tolerance: float = 0.02) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}: {actual:.3f}, 기대 {expected:.3f}±{tolerance}")


def validate_all_scenarios(bundle) -> None:
    invalid = []
    over_28_count = 0
    added_distances = []
    for _, pair in bundle.candidate_pairs.iterrows():
        a_code, b_code = pair[PAIR_A_CODE], pair[PAIR_B_CODE]
        try:
            scenario, grid = run_scenario(
                bundle.master,
                bundle.candidate_pairs,
                bundle.school_points,
                bundle.catchments,
                a_code,
                b_code,
            )
            resource = scenario["resource"]
            access = scenario["accessibility"]
            values = [
                resource["classes_before"],
                resource["classes_after"],
                resource["class_size_before"],
                resource["class_size_after"],
                resource["students_per_teacher_after"],
                resource["students_per_classroom_after"],
                resource["land_per_student_after"],
                access["current_mean_km"],
                access["after_mean_km"],
                access["added_mean_km"],
                access["worsened_pct"],
            ]
            if not grid.empty and all(value is not None and math.isfinite(float(value)) for value in values):
                over_28_count += int(resource["overcrowded_28_after"])
                added_distances.append(access["added_mean_km"])
            else:
                invalid.append((a_code, b_code, "결측 또는 비유한값"))
        except Exception as error:
            invalid.append((a_code, b_code, str(error)))
    if invalid:
        raise AssertionError(f"전체 시나리오 검증 실패 {len(invalid)}건. 첫 사례: {invalid[0]}")
    print("\n전체 후보 시나리오 검증 완료")
    print(f"- 정상 계산: {len(bundle.candidate_pairs):,} / {len(bundle.candidate_pairs):,}건")
    print(f"- 통합 후 28명 참고선 이상: {over_28_count:,}건")
    print(f"- 평균 추가 접근거리 범위: {min(added_distances):+.3f} ~ {max(added_distances):+.3f}km")


def validate_priority_rankings(bundle) -> None:
    """대시보드와 같은 반복 가상점 접근성으로 모든 A별 추천순위를 검증한다."""
    resource_scenarios = build_resource_scenario_table(bundle.master, bundle.candidate_pairs)
    master_lookup = bundle.master.set_index(KEDI)
    recommendations = []
    no_eligible = []
    unconverged = []

    for a_code, pairs in bundle.candidate_pairs.groupby(PAIR_A_CODE, sort=False):
        candidate_codes = tuple(pairs[PAIR_B_CODE].astype(str))
        accessibility, _ = simulate_accessibility_for_candidates(
            bundle.catchments,
            bundle.school_points,
            str(a_code),
            candidate_codes,
            int(master_lookup.loc[str(a_code), STUDENTS]),
        )
        ranking = build_priority_ranking(resource_scenarios, accessibility, str(a_code))
        if len(ranking) != len(candidate_codes):
            raise AssertionError(f"{a_code}: 후보 수와 순위표 행 수가 다릅니다.")
        if set(ranking[PAIR_B_CODE].astype(str)) != set(candidate_codes):
            raise AssertionError(f"{a_code}: 순위표의 후보학교 집합이 다릅니다.")
        if not bool(accessibility["converged"].all()):
            unconverged.append(str(a_code))

        eligible = ranking.loc[ranking["priority_eligible"]].sort_values("review_rank")
        if eligible.empty:
            no_eligible.append(str(a_code))
            continue
        expected_ranks = list(range(1, len(eligible) + 1))
        actual_ranks = eligible["review_rank"].astype(int).tolist()
        if actual_ranks != expected_ranks:
            raise AssertionError(f"{a_code}: 추천순위가 연속적이지 않습니다: {actual_ranks}")
        top = eligible.iloc[0]
        if int(top["pareto_front"]) != 1:
            raise AssertionError(f"{a_code}: 추천 1순위가 파레토 1전면이 아닙니다.")
        if not str(top["priority_label"]).startswith("최선안"):
            raise AssertionError(f"{a_code}: 추천 1순위 표식이 최선안이 아닙니다.")
        if float(top["balance_score"]) < float(eligible["balance_score"].max()) - 1e-9:
            raise AssertionError(f"{a_code}: 추천 1순위가 최고 균형도가 아닙니다.")
        recommendations.append((str(a_code), str(top[PAIR_B_CODE])))

    if not recommendations:
        raise AssertionError("추천 가능한 통합 대상학교가 한 곳도 없습니다.")
    print("\n전체 추천순위 검증 완료")
    print(f"- 후보가 있는 통합 대상학교: {bundle.candidate_pairs[PAIR_A_CODE].nunique():,}개교")
    print(f"- 최선안이 산출된 학교: {len(recommendations):,}개교")
    print(f"- 기본 확인조건 통과 후보가 없는 학교: {len(no_eligible):,}개교")
    if no_eligible:
        no_eligible_names = master_lookup.loc[no_eligible, SCHOOL_NAME].astype(str).tolist()
        print(f"  · 해당 학교: {', '.join(no_eligible_names)}")
    print(f"- 반복 계산이 최대 횟수까지 진행된 학교: {len(unconverged):,}개교")


def main() -> None:
    parser = argparse.ArgumentParser(description="배포 데이터와 대표 시나리오를 검증합니다.")
    parser.add_argument("--all-scenarios", action="store_true", help="1,481개 후보쌍의 교육자원·접근성을 모두 계산")
    args = parser.parse_args()
    bundle = load_bundle()
    report = assert_valid_bundle(bundle)
    scenario, grid = run_scenario(
        bundle.master,
        bundle.candidate_pairs,
        bundle.school_points,
        bundle.catchments,
        KNOWN_A,
        KNOWN_B,
    )
    resource = scenario["resource"]
    access = scenario["accessibility"]
    assert_close("가남→가야 학교간 거리", scenario["pair"]["distance_km"], 0.716, 0.01)
    assert_close("가야 일반학급 수 통합 전", resource["classes_before"], 42, 0.01)
    assert_close("25명 기준 일반학급 수 통합 후", resource["classes_after"], 42, 0.01)
    assert_close("가야 일반학급당 학생수 통합 전", resource["class_size_before"], 22.714, 0.01)
    assert_close("가야 일반학급당 학생수 통합 후", resource["class_size_after"], 23.905, 0.01)
    assert_close("현재 평균 직선거리", access["current_mean_km"], 0.202, 0.03)
    assert_close("통합 후 평균 직선거리", access["after_mean_km"], 0.775, 0.03)
    assert_close("평균 추가 접근거리", access["added_mean_km"], 0.573, 0.03)
    assert_close("접근성 악화 격자 비율", access["worsened_pct"], 100.0, 0.01)
    if len(grid) != 4:
        raise AssertionError(f"가남초 250m 격자 수: {len(grid)}, 기대 4")

    print(report.to_string(index=False))
    print("\n대표 시나리오 검증 완료: 가남초 → 가야초")
    print(f"- 학교간 직선거리: {scenario['pair']['distance_km']:.3f}km")
    print(
        f"- 일반학급 수: 수용학교 {resource['classes_before']} / 두 학교 현재 합 "
        f"{resource['classes_current_sum']} / 25명 기준 {resource['classes_after']}"
    )
    print(f"- 일반학급당 학생수: {resource['class_size_before']:.1f} → {resource['class_size_after']:.1f}")
    print(f"- 평균 추가 접근거리: {access['added_mean_km']:.3f}km")
    print(f"- 접근성 악화 격자: {access['worsened_pct']:.1f}% ({len(grid)}개 격자)")
    if args.all_scenarios:
        validate_all_scenarios(bundle)
        validate_priority_rankings(bundle)


if __name__ == "__main__":
    main()
