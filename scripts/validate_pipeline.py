from __future__ import annotations

import argparse
import math

from src.data_loader import load_bundle
from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.scenario_engine import run_scenario
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
    assert_close("가야 학급당 학생수 통합 전", resource["class_size_before"], 21.8, 0.1)
    assert_close("가야 학급당 학생수 통합 후", resource["class_size_after"], 23.0, 0.1)
    assert_close("현재 평균 직선거리", access["current_mean_km"], 0.202, 0.03)
    assert_close("통합 후 평균 직선거리", access["after_mean_km"], 0.775, 0.03)
    assert_close("평균 추가 접근거리", access["added_mean_km"], 0.573, 0.03)
    assert_close("접근성 악화 격자 비율", access["worsened_pct"], 100.0, 0.01)
    if len(grid) != 4:
        raise AssertionError(f"가남초 250m 격자 수: {len(grid)}, 기대 4")

    print(report.to_string(index=False))
    print("\n대표 시나리오 검증 완료: 가남초 → 가야초")
    print(f"- 학교간 직선거리: {scenario['pair']['distance_km']:.3f}km")
    print(f"- 학급당 학생수: {resource['class_size_before']:.1f} → {resource['class_size_after']:.1f}")
    print(f"- 평균 추가 접근거리: {access['added_mean_km']:.3f}km")
    print(f"- 접근성 악화 격자: {access['worsened_pct']:.1f}% ({len(grid)}개 격자)")
    if args.all_scenarios:
        validate_all_scenarios(bundle)


if __name__ == "__main__":
    main()
