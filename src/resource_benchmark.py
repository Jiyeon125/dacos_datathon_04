from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.schema import (
    CLASSES,
    CLASSROOMS,
    KEDI,
    LAND_AREA,
    SCHOOL_NAME,
    STUDENTS,
    TEACHERS,
    normalize_kedi,
)


RADAR_METRICS = (
    {
        "axis": "학급 여유",
        "column": "class_size_after",
        "raw_label": "학급당 학생 수",
        "unit": "명",
        "higher_is_better": False,
    },
    {
        "axis": "교원 여유",
        "column": "students_per_teacher_after",
        "raw_label": "교원 1인당 학생 수",
        "unit": "명",
        "higher_is_better": False,
    },
    {
        "axis": "교실 여유",
        "column": "students_per_classroom_after",
        "raw_label": "학생/교실",
        "unit": "명",
        "higher_is_better": False,
    },
    {
        "axis": "교지 여유",
        "column": "land_per_student_after",
        "raw_label": "학생 1인당 교지면적",
        "unit": "㎡",
        "higher_is_better": True,
    },
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator.div(denominator.where(denominator.gt(0))).replace([np.inf, -np.inf], np.nan)


def build_resource_scenario_table(master: pd.DataFrame, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    """모든 A-B 후보쌍의 고정자원 가정 결과를 벡터 연산으로 계산한다."""
    lookup = master.copy()
    lookup[KEDI] = normalize_kedi(lookup[KEDI])
    if lookup[KEDI].duplicated().any():
        raise ValueError("학교 마스터의 KEDI가 유일하지 않습니다.")
    lookup = lookup.set_index(KEDI)

    pairs = candidate_pairs.copy()
    pairs[PAIR_A_CODE] = normalize_kedi(pairs[PAIR_A_CODE])
    pairs[PAIR_B_CODE] = normalize_kedi(pairs[PAIR_B_CODE])
    unknown_codes = (set(pairs[PAIR_A_CODE]) | set(pairs[PAIR_B_CODE])) - set(lookup.index)
    if unknown_codes:
        raise KeyError(f"학교 마스터에 없는 후보 코드: {sorted(unknown_codes)}")

    a_students = pd.to_numeric(pairs[PAIR_A_CODE].map(lookup[STUDENTS]), errors="coerce")
    b_students = pd.to_numeric(pairs[PAIR_B_CODE].map(lookup[STUDENTS]), errors="coerce")
    after_students = a_students + b_students
    b_classes = pairs[PAIR_B_CODE].map(lookup[CLASSES])
    b_teachers = pairs[PAIR_B_CODE].map(lookup[TEACHERS])
    b_classrooms = pairs[PAIR_B_CODE].map(lookup[CLASSROOMS])
    b_land = pairs[PAIR_B_CODE].map(lookup[LAND_AREA])

    result = pd.DataFrame(
        {
            PAIR_A_CODE: pairs[PAIR_A_CODE],
            "소규모학교명": pairs[PAIR_A_CODE].map(lookup[SCHOOL_NAME]),
            PAIR_B_CODE: pairs[PAIR_B_CODE],
            "후보학교명": pairs[PAIR_B_CODE].map(lookup[SCHOOL_NAME]),
            "학교간직선거리_km": pd.to_numeric(pairs["학교간직선거리_km"], errors="coerce"),
            "moving_students": a_students,
            "students_before": b_students,
            "students_after": after_students,
            "class_size_after": _safe_divide(after_students, b_classes),
            "students_per_teacher_after": _safe_divide(after_students, b_teachers),
            "students_per_classroom_after": _safe_divide(after_students, b_classrooms),
            "land_per_student_after": _safe_divide(b_land, after_students),
        }
    )
    return result


def comparative_resource_profile(
    scenario_table: pd.DataFrame,
    a_code: str,
    b_code: str,
    *,
    same_a_only: bool,
) -> tuple[pd.DataFrame, int]:
    """선택 시나리오의 지표별 '유리한 방향 백분위'를 비교집단 안에서 계산한다.

    낮을수록 부담이 작은 세 지표는 방향을 뒤집고, 교지면적은 높은 값을
    유리하게 처리한다. 축을 합산한 종합점수는 만들지 않는다.
    """
    a_code, b_code = str(a_code), str(b_code)
    scope = scenario_table.loc[scenario_table[PAIR_A_CODE].eq(a_code)].copy() if same_a_only else scenario_table.copy()
    selected = scope.loc[scope[PAIR_A_CODE].eq(a_code) & scope[PAIR_B_CODE].eq(b_code)]
    if selected.empty:
        raise KeyError(f"비교집단에 없는 시나리오: {a_code} → {b_code}")
    if len(selected) > 1:
        raise ValueError("A-B 후보쌍이 중복되어 있습니다.")
    selected_index = selected.index[0]

    rows: list[dict[str, Any]] = []
    for metric in RADAR_METRICS:
        values = pd.to_numeric(scope[metric["column"]], errors="coerce")
        ranks = values.rank(
            method="average",
            pct=True,
            ascending=bool(metric["higher_is_better"]),
        )
        rows.append(
            {
                "axis": metric["axis"],
                "percentile": float(ranks.loc[selected_index]) * 100 if pd.notna(ranks.loc[selected_index]) else np.nan,
                "raw_value": float(values.loc[selected_index]) if pd.notna(values.loc[selected_index]) else np.nan,
                "raw_label": metric["raw_label"],
                "unit": metric["unit"],
                "valid_n": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows), len(scope)
