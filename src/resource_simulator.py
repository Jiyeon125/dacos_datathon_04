from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.class_formation import simulate_grade_class_formation
from src.schema import (
    CLASSROOMS,
    GENERAL_CLASSROOMS,
    KEDI,
    LAND_AREA,
    SCHOOL_NAME,
    STUDENTS,
    TEACHERS,
    normalize_kedi,
)


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def simulate_resource_change(master: pd.DataFrame, a_code: str, b_code: str) -> dict[str, Any]:
    """A 학생이 B로 이동할 때 학급은 재편성하고 나머지 자원은 B에 고정한다."""
    lookup = master.assign(**{KEDI: normalize_kedi(master[KEDI])}).set_index(KEDI)
    a_code, b_code = str(a_code), str(b_code)
    if a_code not in lookup.index or b_code not in lookup.index:
        raise KeyError("학교 코드가 마스터에 없습니다.")
    if a_code == b_code:
        raise ValueError("A와 B는 서로 다른 학교여야 합니다.")
    a, b = lookup.loc[a_code], lookup.loc[b_code]
    a_students = float(a[STUDENTS])
    before_students = float(b[STUDENTS])
    after_students = before_students + a_students

    formation = simulate_grade_class_formation(a, b)
    classes_before = formation["general_classes_before"]
    classes_after = formation["general_classes_after"]
    class_before = safe_ratio(formation["general_students_before"], classes_before)
    class_after = safe_ratio(formation["general_students_after"], classes_after)
    teacher_before = safe_ratio(before_students, b[TEACHERS])
    teacher_after = safe_ratio(after_students, b[TEACHERS])
    classroom_before = safe_ratio(before_students, b[CLASSROOMS])
    classroom_after = safe_ratio(after_students, b[CLASSROOMS])
    land_before = safe_ratio(b[LAND_AREA], before_students)
    land_after = safe_ratio(b[LAND_AREA], after_students)
    general_classrooms = pd.to_numeric(b.get(GENERAL_CLASSROOMS), errors="coerce")
    classroom_gap = None if pd.isna(general_classrooms) else int(classes_after - general_classrooms)
    return {
        "a_code": a_code,
        "a_name": a[SCHOOL_NAME],
        "b_code": b_code,
        "b_name": b[SCHOOL_NAME],
        "moving_students": int(a_students),
        "students_before": int(before_students),
        "students_after": int(after_students),
        "classes_before": int(classes_before),
        "classes_current_sum": int(formation["general_classes_current_sum"]),
        "classes_after": int(classes_after),
        "classes_delta": int(formation["general_classes_delta_vs_b"]),
        "classes_delta_vs_current_sum": int(formation["general_classes_delta_vs_current_sum"]),
        "class_size_before": class_before,
        "class_size_after": class_after,
        "class_size_delta": _delta(class_before, class_after),
        "students_per_teacher_before": teacher_before,
        "students_per_teacher_after": teacher_after,
        "students_per_teacher_delta": _delta(teacher_before, teacher_after),
        "students_per_classroom_before": classroom_before,
        "students_per_classroom_after": classroom_after,
        "students_per_classroom_delta": _delta(classroom_before, classroom_after),
        "land_per_student_before": land_before,
        "land_per_student_after": land_after,
        "land_per_student_delta": _delta(land_before, land_after),
        "overcrowded_28_before": bool(class_before is not None and class_before >= 28),
        "overcrowded_28_after": bool(class_after is not None and class_after >= 28),
        "class_rule_year": formation["rule_year"],
        "class_rule_capacity": formation["students_per_class"],
        "class_rule_status": formation["rule_status"],
        "class_rule_label": formation["rule_label"],
        "class_rule_source_urls": formation["source_urls"],
        "grade_class_plan": formation["grade_plan"],
        "general_students_before": formation["general_students_before"],
        "general_students_after": formation["general_students_after"],
        "special_students_current_sum": formation["special_students_current_sum"],
        "special_classes_current_sum": formation["special_classes_current_sum"],
        "general_classrooms_b": None if pd.isna(general_classrooms) else int(general_classrooms),
        "general_classroom_gap": classroom_gap,
        "general_classroom_shortage": bool(classroom_gap is not None and classroom_gap > 0),
        "assumption": (
            "2025년 4월 학년별 일반학생을 합산해 부산 초등 학생배치지표 25명으로 일반학급을 재편성하고, "
            "교원·교실·교지는 수용학교의 현재 규모를 유지"
        ),
    }


def resource_comparison_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("학생 수", result["students_before"], result["students_after"], result["moving_students"], "명"),
        ("일반학급 수", result["classes_before"], result["classes_after"], result["classes_delta"], "학급"),
        ("일반학급당 학생 수", result["class_size_before"], result["class_size_after"], result["class_size_delta"], "명"),
        ("교원 1인당 학생 수", result["students_per_teacher_before"], result["students_per_teacher_after"], result["students_per_teacher_delta"], "명"),
        ("학생/교실", result["students_per_classroom_before"], result["students_per_classroom_after"], result["students_per_classroom_delta"], "명"),
        ("학생 1인당 교지면적", result["land_per_student_before"], result["land_per_student_after"], result["land_per_student_delta"], "㎡"),
    ]
    table = pd.DataFrame(rows, columns=["지표", "통합 전", "통합 후", "변화", "단위"])
    for column in ["통합 전", "통합 후", "변화"]:
        table[column] = table[column].map(lambda value: np.nan if value is None else round(float(value), 2))
    return table


def grade_class_comparison_table(result: dict[str, Any]) -> pd.DataFrame:
    table = pd.DataFrame(result["grade_class_plan"]).rename(
        columns={
            "grade": "학년",
            "a_general_students": "통합 대상 학생",
            "b_general_students": "수용학교 학생",
            "combined_general_students": "통합 후 학생",
            "a_current_general_classes": "통합 대상 현재 학급",
            "b_current_general_classes": "수용학교 현재 학급",
            "current_general_classes_sum": "현재 학급 합",
            "required_general_classes": "통합 후 필요 학급",
            "class_change_vs_current_sum": "학급 통합효과",
            "students_per_required_class": "편성 후 학급당 학생",
        }
    )
    columns = [
        "학년",
        "통합 대상 학생",
        "수용학교 학생",
        "통합 후 학생",
        "통합 대상 현재 학급",
        "수용학교 현재 학급",
        "현재 학급 합",
        "통합 후 필요 학급",
        "학급 통합효과",
        "편성 후 학급당 학생",
    ]
    table = table[columns].copy()
    table["학년"] = table["학년"].map(lambda value: f"{int(value)}학년")
    table["학급 통합효과"] = table["학급 통합효과"].map(lambda value: f"{int(value):+d}")
    table["편성 후 학급당 학생"] = table["편성 후 학급당 학생"].round(1)
    return table
