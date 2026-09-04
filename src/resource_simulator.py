from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.schema import CLASSES, CLASSROOMS, KEDI, LAND_AREA, SCHOOL_NAME, STUDENTS, TEACHERS, normalize_kedi


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def simulate_resource_change(master: pd.DataFrame, a_code: str, b_code: str) -> dict[str, Any]:
    """A 학생 전원이 B로 이동하고 B의 학급·교원·교실·교지는 고정된 가정이다."""
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

    class_before = safe_ratio(before_students, b[CLASSES])
    class_after = safe_ratio(after_students, b[CLASSES])
    teacher_before = safe_ratio(before_students, b[TEACHERS])
    teacher_after = safe_ratio(after_students, b[TEACHERS])
    classroom_before = safe_ratio(before_students, b[CLASSROOMS])
    classroom_after = safe_ratio(after_students, b[CLASSROOMS])
    land_before = safe_ratio(b[LAND_AREA], before_students)
    land_after = safe_ratio(b[LAND_AREA], after_students)
    return {
        "a_code": a_code,
        "a_name": a[SCHOOL_NAME],
        "b_code": b_code,
        "b_name": b[SCHOOL_NAME],
        "moving_students": int(a_students),
        "students_before": int(before_students),
        "students_after": int(after_students),
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
        "assumption": "A 학생 전원이 B로 이동하며 B의 학급·교원·교실·교지 규모는 그대로 유지",
    }


def resource_comparison_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("학생 수", result["students_before"], result["students_after"], result["moving_students"], "명"),
        ("학급당 학생 수", result["class_size_before"], result["class_size_after"], result["class_size_delta"], "명"),
        ("교원 1인당 학생 수", result["students_per_teacher_before"], result["students_per_teacher_after"], result["students_per_teacher_delta"], "명"),
        ("학생/교실", result["students_per_classroom_before"], result["students_per_classroom_after"], result["students_per_classroom_delta"], "명"),
        ("학생 1인당 교지면적", result["land_per_student_before"], result["land_per_student_after"], result["land_per_student_delta"], "㎡"),
    ]
    table = pd.DataFrame(rows, columns=["지표", "통합 전", "통합 후", "변화", "단위"])
    for column in ["통합 전", "통합 후", "변화"]:
        table[column] = table[column].map(lambda value: np.nan if value is None else round(float(value), 2))
    return table
