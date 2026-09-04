import pandas as pd

from src.class_formation import (
    general_class_column,
    general_student_column,
    special_class_column,
    special_student_column,
)
from src.resource_simulator import grade_class_comparison_table, simulate_resource_change
from src.schema import (
    CLASSES,
    CLASSROOMS,
    DISTRICT,
    GENERAL_CLASSROOMS,
    KEDI,
    LAND_AREA,
    SCHOOL_NAME,
    SMALL_FLAG,
    STUDENTS,
    TEACHERS,
)


def _school_row(code: str, name: str, students: int, classes: int, teachers: int, classrooms: int, land: int, small: bool, grade_students: list[int], grade_classes: list[int]) -> dict:
    row = {
        KEDI: code,
        SCHOOL_NAME: name,
        DISTRICT: "갑구",
        STUDENTS: students,
        CLASSES: classes,
        TEACHERS: teachers,
        CLASSROOMS: classrooms,
        GENERAL_CLASSROOMS: classrooms,
        LAND_AREA: land,
        SMALL_FLAG: small,
    }
    for grade, (grade_student, grade_class) in enumerate(zip(grade_students, grade_classes), start=1):
        row[general_student_column(grade)] = grade_student
        row[general_class_column(grade)] = grade_class
        row[special_student_column(grade)] = 0
        row[special_class_column(grade)] = 0
    return row


def test_grade_based_class_formation_and_fixed_other_resources():
    master = pd.DataFrame(
        [
            _school_row("A", "A초", 80, 6, 12, 8, 4000, True, [8, 10, 11, 14, 17, 20], [1, 1, 1, 1, 1, 1]),
            _school_row("B", "B초", 350, 16, 27, 20, 11200, False, [55, 56, 57, 58, 60, 64], [2, 2, 3, 3, 3, 3]),
        ]
    )
    result = simulate_resource_change(master, "A", "B")
    assert result["students_after"] == 430
    assert result["classes_before"] == 16
    assert result["classes_current_sum"] == 22
    assert result["classes_after"] == 20
    assert result["classes_delta_vs_current_sum"] == -2
    assert result["class_rule_capacity"] == 25
    assert round(result["class_size_before"], 1) == 21.9
    assert round(result["class_size_after"], 1) == 21.5
    assert round(result["students_per_teacher_after"], 1) == 15.9
    assert result["teacher_current_b"] == 27
    assert result["teacher_current_sum"] == 39
    assert result["teacher_model_input_classes"] == 20
    assert round(result["teacher_reference_estimate"], 1) == 30.8
    assert result["teacher_reference_range_low"] < result["teacher_reference_estimate"]
    assert result["teacher_reference_range_high"] > result["teacher_reference_estimate"]
    assert round(result["students_per_classroom_after"], 1) == 21.5
    assert round(result["land_per_student_after"], 1) == 26.0
    assert [row["required_general_classes"] for row in result["grade_class_plan"]] == [3, 3, 3, 3, 4, 4]
    assert list(grade_class_comparison_table(result)["학급 통합효과"]) == ["+0", "+0", "-1", "-1", "+0", "+0"]
