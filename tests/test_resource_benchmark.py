import pytest

from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.data_loader import load_bundle
from src.resource_benchmark import build_resource_scenario_table, comparative_resource_profile
from src.resource_simulator import simulate_resource_change


A_CODE = "213021106"
B_CODE = "213021124"


def test_resource_scenario_table_matches_single_scenario_engine():
    bundle = load_bundle()
    table = build_resource_scenario_table(bundle.master, bundle.candidate_pairs)
    single = simulate_resource_change(bundle.master, A_CODE, B_CODE)
    row = table.loc[table[PAIR_A_CODE].eq(A_CODE) & table[PAIR_B_CODE].eq(B_CODE)].iloc[0]

    assert len(table) == len(bundle.candidate_pairs) == 1481
    assert not table.duplicated([PAIR_A_CODE, PAIR_B_CODE]).any()
    assert row["classes_before"] == pytest.approx(single["classes_before"])
    assert row["classes_after"] == pytest.approx(single["classes_after"])
    assert row["classes_delta_vs_current_sum"] == pytest.approx(single["classes_delta_vs_current_sum"])
    assert row["class_size_after"] == pytest.approx(single["class_size_after"])
    assert row["students_per_teacher_after"] == pytest.approx(single["students_per_teacher_after"])
    assert row["teacher_current_b"] == pytest.approx(single["teacher_current_b"])
    assert row["teacher_current_sum"] == pytest.approx(single["teacher_current_sum"])
    assert row["teacher_model_input_classes"] == pytest.approx(single["teacher_model_input_classes"])
    assert row["teacher_reference_estimate"] == pytest.approx(single["teacher_reference_estimate"])
    assert row["teacher_reference_range_low"] == pytest.approx(single["teacher_reference_range_low"])
    assert row["teacher_reference_range_high"] == pytest.approx(single["teacher_reference_range_high"])
    assert row["students_per_classroom_after"] == pytest.approx(single["students_per_classroom_after"])
    assert row["land_per_student_after"] == pytest.approx(single["land_per_student_after"])


def test_comparative_profiles_use_same_a_and_all_scenario_scopes():
    bundle = load_bundle()
    table = build_resource_scenario_table(bundle.master, bundle.candidate_pairs)

    same_a, same_a_size = comparative_resource_profile(table, A_CODE, B_CODE, same_a_only=True)
    all_cases, all_size = comparative_resource_profile(table, A_CODE, B_CODE, same_a_only=False)

    assert same_a_size == int(table[PAIR_A_CODE].eq(A_CODE).sum()) == 32
    assert all_size == len(table) == 1481
    assert len(same_a) == len(all_cases) == 4
    assert same_a["percentile"].between(0, 100).all()
    assert all_cases["percentile"].between(0, 100).all()


def test_lower_burden_receives_higher_favorable_percentile():
    bundle = load_bundle()
    table = build_resource_scenario_table(bundle.master, bundle.candidate_pairs)
    scope = table.loc[table[PAIR_A_CODE].eq(A_CODE)].dropna(subset=["class_size_after"])
    low_burden = scope.nsmallest(1, "class_size_after").iloc[0]
    high_burden = scope.nlargest(1, "class_size_after").iloc[0]

    low_profile, _ = comparative_resource_profile(
        table,
        A_CODE,
        low_burden[PAIR_B_CODE],
        same_a_only=True,
    )
    high_profile, _ = comparative_resource_profile(
        table,
        A_CODE,
        high_burden[PAIR_B_CODE],
        same_a_only=True,
    )

    low_percentile = low_profile.loc[low_profile["axis"].eq("일반학급 여유"), "percentile"].iloc[0]
    high_percentile = high_profile.loc[high_profile["axis"].eq("일반학급 여유"), "percentile"].iloc[0]
    assert low_percentile > high_percentile
