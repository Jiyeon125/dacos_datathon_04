from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE


RESOURCE_PRIORITY_METRICS: tuple[dict[str, Any], ...] = (
    {
        "column": "class_size_after",
        "percentile_column": "class_size_favorable_pct",
        "label": "일반학급당 학생 수",
        "higher_is_better": False,
    },
    {
        "column": "students_per_teacher_after",
        "percentile_column": "teacher_load_favorable_pct",
        "label": "교원 1인당 학생 수",
        "higher_is_better": False,
    },
    {
        "column": "students_per_classroom_after",
        "percentile_column": "classroom_load_favorable_pct",
        "label": "학생/교실",
        "higher_is_better": False,
    },
    {
        "column": "land_per_student_after",
        "percentile_column": "land_favorable_pct",
        "label": "학생 1인당 교지면적",
        "higher_is_better": True,
    },
)

ACCESS_PRIORITY_METRICS: tuple[dict[str, Any], ...] = (
    {
        "column": "added_mean_km",
        "percentile_column": "mean_access_favorable_pct",
        "label": "평균 추가 접근거리",
        "higher_is_better": False,
    },
    {
        "column": "added_max_km",
        "percentile_column": "max_access_favorable_pct",
        "label": "최대 추가 접근거리",
        "higher_is_better": False,
    },
)

PRIORITY_METRICS = (*RESOURCE_PRIORITY_METRICS, *ACCESS_PRIORITY_METRICS)


def _favorable_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for metric in PRIORITY_METRICS:
        values = pd.to_numeric(result[metric["column"]], errors="coerce")
        result[metric["percentile_column"]] = values.rank(
            method="average",
            pct=True,
            ascending=bool(metric["higher_is_better"]),
        ) * 100
    return result


def _pareto_fronts(values: pd.DataFrame) -> pd.Series:
    """모든 열이 클수록 유리한 값에서 비지배 전면 번호를 계산한다."""
    fronts = pd.Series(pd.NA, index=values.index, dtype="Int64")
    remaining = list(values.index)
    front_number = 1
    while remaining:
        front: list[Any] = []
        for index in remaining:
            candidate = values.loc[index].to_numpy(dtype=float)
            dominated = False
            for other_index in remaining:
                if other_index == index:
                    continue
                other = values.loc[other_index].to_numpy(dtype=float)
                if np.all(other >= candidate) and np.any(other > candidate):
                    dominated = True
                    break
            if not dominated:
                front.append(index)
        if not front:
            raise RuntimeError("파레토 전면을 계산할 수 없습니다.")
        fronts.loc[front] = front_number
        remaining = [index for index in remaining if index not in front]
        front_number += 1
    return fronts


def _constraint_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["priority_metrics_complete"]):
        reasons.append("핵심 지표 자료 없음")
    if not bool(row["general_classroom_known"]):
        reasons.append("일반교실 자료 없음")
    if bool(row["overcrowded_reference_after"]):
        reasons.append("28명 과밀 참고선 이상")
    if bool(row["general_classroom_known"]) and bool(row["general_classroom_shortage"]):
        reasons.append("필요 일반학급이 일반교실 수 초과")
    return " · ".join(reasons) if reasons else "기본 확인조건 통과"


def build_priority_ranking(
    resource_scenarios: pd.DataFrame,
    accessibility_scenarios: pd.DataFrame,
    a_code: str,
) -> pd.DataFrame:
    """같은 통합 대상학교의 수용 후보에 비가중 다기준 검토순위를 만든다.

    28명 과밀 참고선과 일반교실 부족을 먼저 확인한 뒤, 통과 후보를
    최약 지표부터 차례로 비교(leximin)하고 파레토 전면을 함께 표시한다.
    회귀 교원 참고값은 공식 배치값이 아니므로 순위 산정에 사용하지 않는다.
    """
    a_code = str(a_code)
    resources = resource_scenarios.loc[resource_scenarios[PAIR_A_CODE].astype(str).eq(a_code)].copy()
    access = accessibility_scenarios.copy()
    access[PAIR_B_CODE] = access[PAIR_B_CODE].astype(str)
    if resources.empty:
        return resources
    if resources[PAIR_B_CODE].duplicated().any() or access[PAIR_B_CODE].duplicated().any():
        raise ValueError("후보학교 코드가 중복되어 있습니다.")

    access_columns = [
        PAIR_B_CODE,
        "added_mean_km",
        "added_median_km",
        "added_max_km",
        "worsened_pct",
        "converged",
    ]
    missing_access = set(access_columns) - set(access.columns)
    if missing_access:
        raise KeyError(f"접근성 순위 입력열이 없습니다: {sorted(missing_access)}")
    ranked = resources.merge(access[access_columns], on=PAIR_B_CODE, how="left", validate="one_to_one")

    metric_columns = [metric["column"] for metric in PRIORITY_METRICS]
    percentile_columns = [metric["percentile_column"] for metric in PRIORITY_METRICS]
    resource_percentiles = [metric["percentile_column"] for metric in RESOURCE_PRIORITY_METRICS]
    access_percentiles = [metric["percentile_column"] for metric in ACCESS_PRIORITY_METRICS]
    leximin_columns = [f"_leximin_{number}" for number in range(1, len(percentile_columns) + 1)]
    ranked["priority_metrics_complete"] = ranked[metric_columns].notna().all(axis=1)
    ranked["overcrowded_reference_after"] = pd.to_numeric(
        ranked["class_size_after"], errors="coerce"
    ).ge(28)
    classroom_shortage = ranked["general_classroom_shortage"].astype("boolean")
    ranked["general_classroom_known"] = classroom_shortage.notna()
    ranked["priority_eligible"] = (
        ranked["priority_metrics_complete"]
        & ~ranked["overcrowded_reference_after"]
        & ranked["general_classroom_known"]
        & classroom_shortage.eq(False).fillna(False)
    )
    ranked["constraint_reason"] = ranked.apply(_constraint_reason, axis=1)
    for column in percentile_columns:
        ranked[column] = np.nan
    ranked["resource_all_above_median"] = False
    ranked["resource_balance_pct"] = np.nan
    ranked["access_balance_pct"] = np.nan
    ranked["balance_score"] = np.nan
    for column in leximin_columns:
        ranked[column] = np.nan
    ranked["pareto_front"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked["review_rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")

    eligible_index = ranked.index[ranked["priority_eligible"]]
    if len(eligible_index):
        eligible_scored = _favorable_percentiles(ranked.loc[eligible_index])
        ranked.loc[eligible_index, percentile_columns] = eligible_scored[percentile_columns]
        ranked.loc[eligible_index, "resource_all_above_median"] = (
            eligible_scored[resource_percentiles].ge(50).all(axis=1)
        )
        ranked.loc[eligible_index, "resource_balance_pct"] = eligible_scored[resource_percentiles].min(axis=1)
        ranked.loc[eligible_index, "access_balance_pct"] = eligible_scored[access_percentiles].min(axis=1)
        sorted_percentiles = np.sort(eligible_scored[percentile_columns].to_numpy(dtype=float), axis=1)
        ranked.loc[eligible_index, leximin_columns] = sorted_percentiles
        ranked.loc[eligible_index, "balance_score"] = sorted_percentiles[:, 0]
        ranked.loc[eligible_index, "pareto_front"] = _pareto_fronts(
            ranked.loc[eligible_index, percentile_columns]
        )
        eligible_order = ranked.loc[eligible_index].sort_values(
            [*leximin_columns,
                "pareto_front",
                "added_mean_km",
                "학교간직선거리_km",
                "후보학교명",
            ],
            ascending=[False] * len(leximin_columns) + [True, True, True, True],
            kind="stable",
        ).index
        ranked.loc[eligible_order, "review_rank"] = range(1, len(eligible_order) + 1)

    def label(row: pd.Series) -> str:
        if not bool(row["priority_eligible"]):
            return "제약 확인"
        if int(row["review_rank"]) == 1 and bool(row["resource_all_above_median"]):
            return "최선안·자원 전 지표 상대우수"
        if int(row["review_rank"]) == 1:
            return "최선안"
        if int(row["review_rank"]) <= 3:
            return "상위 대안"
        if int(row["pareto_front"]) == 1:
            return "파레토 우선후보"
        return f"파레토 {int(row['pareto_front'])}단계 후보"

    ranked["priority_label"] = ranked.apply(label, axis=1)
    ranked["display_order"] = ranked["review_rank"].fillna(10_000).astype(int)
    return ranked.sort_values(
        ["display_order", "학교간직선거리_km", "후보학교명"],
        kind="stable",
    ).drop(columns=["display_order", *leximin_columns]).reset_index(drop=True)
