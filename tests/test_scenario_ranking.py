import pandas as pd

from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.scenario_ranking import build_priority_ranking


def _resource_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                PAIR_A_CODE: "A",
                PAIR_B_CODE: "B1",
                "후보학교명": "균형학교",
                "학교간직선거리_km": 1.0,
                "class_size_after": 20.0,
                "students_per_teacher_after": 11.0,
                "students_per_classroom_after": 10.0,
                "land_per_student_after": 40.0,
                "general_classroom_shortage": False,
            },
            {
                PAIR_A_CODE: "A",
                PAIR_B_CODE: "B2",
                "후보학교명": "접근학교",
                "학교간직선거리_km": 0.7,
                "class_size_after": 22.0,
                "students_per_teacher_after": 13.0,
                "students_per_classroom_after": 12.0,
                "land_per_student_after": 30.0,
                "general_classroom_shortage": False,
            },
            {
                PAIR_A_CODE: "A",
                PAIR_B_CODE: "B3",
                "후보학교명": "열위학교",
                "학교간직선거리_km": 1.4,
                "class_size_after": 23.0,
                "students_per_teacher_after": 14.0,
                "students_per_classroom_after": 13.0,
                "land_per_student_after": 25.0,
                "general_classroom_shortage": False,
            },
            {
                PAIR_A_CODE: "A",
                PAIR_B_CODE: "B4",
                "후보학교명": "과밀학교",
                "학교간직선거리_km": 0.5,
                "class_size_after": 29.0,
                "students_per_teacher_after": 10.0,
                "students_per_classroom_after": 9.0,
                "land_per_student_after": 50.0,
                "general_classroom_shortage": False,
            },
        ]
    )


def _access_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {PAIR_B_CODE: "B1", "added_mean_km": 0.8, "added_median_km": 0.7, "added_max_km": 1.4, "worsened_pct": 90.0, "converged": True},
            {PAIR_B_CODE: "B2", "added_mean_km": 0.4, "added_median_km": 0.3, "added_max_km": 0.9, "worsened_pct": 80.0, "converged": True},
            {PAIR_B_CODE: "B3", "added_mean_km": 1.2, "added_median_km": 1.1, "added_max_km": 1.9, "worsened_pct": 100.0, "converged": True},
            {PAIR_B_CODE: "B4", "added_mean_km": 0.2, "added_median_km": 0.2, "added_max_km": 0.6, "worsened_pct": 70.0, "converged": True},
        ]
    )


def test_priority_ranking_filters_constraints_and_keeps_tradeoff_front():
    ranking = build_priority_ranking(_resource_rows(), _access_rows(), "A").set_index(PAIR_B_CODE)

    assert ranking.loc["B1", "priority_eligible"]
    assert ranking.loc["B2", "priority_eligible"]
    assert ranking.loc["B1", "pareto_front"] == 1
    assert ranking.loc["B2", "pareto_front"] == 1
    assert ranking.loc["B3", "pareto_front"] > 1
    assert not ranking.loc["B4", "priority_eligible"]
    assert "28명" in ranking.loc["B4", "constraint_reason"]
    assert pd.isna(ranking.loc["B4", "balance_score"])
    assert ranking.loc[["B1", "B2", "B3"], "balance_score"].min() >= (100 / 3) - 1e-9
    assert int(ranking.loc["B1", "review_rank"]) == 1
    assert set(ranking.loc[["B1", "B2", "B3"], "review_rank"].astype(int)) == {1, 2, 3}


def test_priority_ranking_is_limited_to_selected_source_school():
    resources = pd.concat([
        _resource_rows(),
        _resource_rows().assign(**{PAIR_A_CODE: "OTHER"}),
    ], ignore_index=True)
    ranking = build_priority_ranking(resources, _access_rows(), "A")
    assert set(ranking[PAIR_A_CODE]) == {"A"}
