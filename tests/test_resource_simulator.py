import pandas as pd

from src.resource_simulator import simulate_resource_change
from src.schema import CLASSES, CLASSROOMS, DISTRICT, KEDI, LAND_AREA, SCHOOL_NAME, SMALL_FLAG, STUDENTS, TEACHERS


def test_fixed_resource_scenario_calculates_before_after():
    master = pd.DataFrame(
        [
            {KEDI: "A", SCHOOL_NAME: "A초", DISTRICT: "갑구", STUDENTS: 80, CLASSES: 6, TEACHERS: 12, CLASSROOMS: 8, LAND_AREA: 4000, SMALL_FLAG: True},
            {KEDI: "B", SCHOOL_NAME: "B초", DISTRICT: "갑구", STUDENTS: 350, CLASSES: 16, TEACHERS: 27, CLASSROOMS: 20, LAND_AREA: 11200, SMALL_FLAG: False},
        ]
    )
    result = simulate_resource_change(master, "A", "B")
    assert result["students_after"] == 430
    assert round(result["class_size_before"], 1) == 21.9
    assert round(result["class_size_after"], 1) == 26.9
    assert round(result["students_per_teacher_after"], 1) == 15.9
    assert round(result["students_per_classroom_after"], 1) == 21.5
    assert round(result["land_per_student_after"], 1) == 26.0

