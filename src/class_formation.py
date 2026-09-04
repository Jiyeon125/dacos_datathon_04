from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import BUSAN_ELEMENTARY_CLASS_FORMATION_2025


GRADES = tuple(range(1, 7))


def general_student_column(grade: int) -> str:
    return f"일반학생수_20250401_{grade}학년"


def general_class_column(grade: int) -> str:
    return f"일반학급수_20250401_{grade}학년"


def special_student_column(grade: int) -> str:
    return f"특수학생수_20250401_{grade}학년"


def special_class_column(grade: int) -> str:
    return f"특수학급수_20250401_{grade}학년"


def _required_number(school: pd.Series, column: str) -> float:
    if column not in school.index:
        raise ValueError(f"학급 재편성 필수 컬럼 누락: {column}")
    value = pd.to_numeric(school[column], errors="coerce")
    if pd.isna(value):
        raise ValueError(f"학급 재편성 값 누락: {column}")
    return float(value)


def simulate_grade_class_formation(
    a_school: pd.Series,
    b_school: pd.Series,
    *,
    students_per_class: int | None = None,
) -> dict[str, Any]:
    """2025 부산 초등 학생배치지표로 학년별 일반학급을 다시 편성한다.

    특수학급은 일반학급 계산에 섞지 않고 현재 A+B 규모만 별도로 반환한다.
    """
    rule = BUSAN_ELEMENTARY_CLASS_FORMATION_2025
    capacity = int(students_per_class or rule["students_per_class"])
    if capacity <= 0:
        raise ValueError("학급당 기준인원은 1명 이상이어야 합니다.")

    rows: list[dict[str, Any]] = []
    for grade in GRADES:
        a_students = _required_number(a_school, general_student_column(grade))
        b_students = _required_number(b_school, general_student_column(grade))
        a_classes = _required_number(a_school, general_class_column(grade))
        b_classes = _required_number(b_school, general_class_column(grade))
        a_special_students = _required_number(a_school, special_student_column(grade))
        b_special_students = _required_number(b_school, special_student_column(grade))
        a_special_classes = _required_number(a_school, special_class_column(grade))
        b_special_classes = _required_number(b_school, special_class_column(grade))
        combined_students = a_students + b_students
        required_classes = math.ceil(combined_students / capacity) if combined_students > 0 else 0
        rows.append(
            {
                "grade": grade,
                "a_general_students": int(a_students),
                "b_general_students": int(b_students),
                "combined_general_students": int(combined_students),
                "a_current_general_classes": int(a_classes),
                "b_current_general_classes": int(b_classes),
                "current_general_classes_sum": int(a_classes + b_classes),
                "required_general_classes": int(required_classes),
                "class_change_vs_current_sum": int(required_classes - a_classes - b_classes),
                "students_per_required_class": combined_students / required_classes if required_classes else None,
                "special_students_current_sum": int(a_special_students + b_special_students),
                "special_classes_current_sum": int(a_special_classes + b_special_classes),
            }
        )

    plan = pd.DataFrame(rows)
    return {
        "rule_year": int(rule["year"]),
        "students_per_class": capacity,
        "rule_status": rule["status"],
        "rule_label": rule["label"],
        "source_urls": list(rule["source_urls"]),
        "grade_plan": rows,
        "general_students_before": int(plan["b_general_students"].sum()),
        "general_students_after": int(plan["combined_general_students"].sum()),
        "general_classes_before": int(plan["b_current_general_classes"].sum()),
        "general_classes_current_sum": int(plan["current_general_classes_sum"].sum()),
        "general_classes_after": int(plan["required_general_classes"].sum()),
        "general_classes_delta_vs_b": int(
            plan["required_general_classes"].sum() - plan["b_current_general_classes"].sum()
        ),
        "general_classes_delta_vs_current_sum": int(
            plan["required_general_classes"].sum() - plan["current_general_classes_sum"].sum()
        ),
        "special_students_current_sum": int(plan["special_students_current_sum"].sum()),
        "special_classes_current_sum": int(plan["special_classes_current_sum"].sum()),
    }
